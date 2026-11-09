import torch
import torch.nn.functional as F
from trainer.utils import compute_dpo_loss
from trainer.unlearn.npo import NPO


class AdvNPO(NPO):
    def __init__(
        self,
        perturb_layer: int = 8,
        epsilon: float = 3.0,
        adv_steps: int = 3,
        random_init: bool = False,
        std_normalization: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.perturb_layer = perturb_layer
        self.epsilon = epsilon
        self.adv_steps = adv_steps
        self.random_init = random_init
        self.std_normalization = std_normalization

    def _get_decoder_layers(self, model):
        m = model
        while hasattr(m, "module"):
            m = m.module
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "layers"):
            return m.layers
        raise AttributeError("Cannot locate decoder layers on model.")

    def _inject_hook(self, model, perturbation):
        layers = self._get_decoder_layers(model)
        target_layer = layers[self.perturb_layer]

        def hook(module, args):
            hidden = args[0]
            args_list = list(args)
            args_list[0] = hidden + perturbation.to(hidden.device)
            return tuple(args_list)

        handle = target_layer.register_forward_pre_hook(hook)
        return handle

    def _get_latents(self, model, inputs):
        latents_store = {}
        layers = self._get_decoder_layers(model)
        target_layer = layers[self.perturb_layer]

        def capture_hook(module, args):
            latents_store["latents"] = args[0].detach().clone()

        handle = target_layer.register_forward_pre_hook(capture_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        return latents_store["latents"]

    def adv_attack(self, model, inputs):
        was_training = model.training
        model.eval()

        latents_orig = self._get_latents(model, inputs)
        batch_size = latents_orig.shape[0]
        min_act = latents_orig.min()
        max_act = latents_orig.max()

        if self.std_normalization:
            act_std = torch.std(latents_orig, dim=0, keepdim=False)
            mean_std = act_std.mean()
            normalization = (act_std / mean_std) + (
                torch.mean(torch.abs(latents_orig)) / 10
            )
        else:
            normalization = 1.0

        def project_clip(delta):
            norms = torch.norm(delta.reshape(batch_size, -1), p=2, dim=1)
            factor = self.epsilon / norms
            factor = torch.min(factor, torch.ones_like(norms))
            delta = delta * factor.view(batch_size, 1, 1)
            adv = torch.clamp(
                latents_orig + delta * normalization, min=min_act, max=max_act
            ).detach()
            return ((adv - latents_orig) / normalization).detach()

        if self.random_init or self.adv_steps == 0:
            delta = torch.empty_like(latents_orig).normal_()
            norms = torch.norm(delta.reshape(batch_size, -1), p=2, dim=1)
            r = torch.zeros(batch_size, device=delta.device).uniform_(0, 1)
            delta = delta * (r / norms).view(batch_size, 1, 1) * self.epsilon
        else:
            delta = torch.zeros_like(latents_orig)

        delta = project_clip(delta)
        stepsize = self.epsilon / max(1, self.adv_steps)

        for step in range(self.adv_steps):
            delta = delta.detach().requires_grad_(True)
            perturb = delta * normalization

            handle = self._inject_hook(model, perturb)
            # モデルパラメータのgradを止めてperturbationだけgradを流す
            with torch.no_grad():
                pass  # no-opでコンテキスト確認
            outputs = model(**inputs)
            handle.remove()

            logits = outputs.logits
            labels = inputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fn(shift_logits.transpose(-1, -2), shift_labels).sum(dim=-1)
            loss = token_loss.mean()

            # パラメータのgradは計算せずdeltaのgradだけ取る
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

            grad_norms = torch.norm(grad.reshape(batch_size, -1), p=2, dim=1) + 1e-6
            grad = grad / grad_norms.view(batch_size, 1, 1)
            delta = project_clip(delta.detach() - stepsize * grad)

        if was_training:
            model.train()

        return (delta * normalization).detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        forget_inputs_clean = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        if self.epsilon > 0 and model.training:
            perturbation = self.adv_attack(model, forget_inputs_clean)
            handle = self._inject_hook(model, perturbation)
        else:
            handle = None

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs_clean,
            beta=self.beta,
        )

        if handle is not None:
            handle.remove()

        retain_inputs = inputs["retain"]
        retain_inputs_clean = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(
            model=model, retain_inputs=retain_inputs_clean
        )

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
