from collections import OrderedDict

import torch
import torch.nn as nn

from .modeling_causal_vae import ActionVQVAE, ActionVQVAEPE


class ActionVQVAELossWrapper(nn.Module):
    """
    The causal action vqvae training and inference running wrapper
    """

    def __init__(
        self,
        model_path,
        freeze=False,
        checkpoint_path=None,
        use_action_type_pe=False,
        use_time_pe=False,
        resume=False,
        is_eval=False,
        **kwargs,
    ):
        super().__init__()

        if use_action_type_pe and use_time_pe:
            self.vqvae = ActionVQVAEPE.from_config(f"{model_path}/config_action_type_pe_time_pe.json")
        elif use_action_type_pe:
            self.vqvae = ActionVQVAEPE.from_config(f"{model_path}/config_action_type_pe.json")
        elif use_time_pe:
            self.vqvae = ActionVQVAEPE.from_config(f"{model_path}/config_time_pe.json")
        else:
            self.vqvae = ActionVQVAE.from_config(f"{model_path}/config.json")
        self.token_num = self.vqvae.vqvae_groups

        if resume:
            assert checkpoint_path is not None, "resume mode must provide checkpoint path"
            self.load_checkpoint(checkpoint_path)
        elif checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        if is_eval:
            self.vqvae.encoder.eval()
            self.vqvae.decoder.eval()
            self.vqvae.vq_layer.eval()

        if freeze:
            for parameter in self.vqvae.encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.vqvae.decoder.parameters():
                parameter.requires_grad = False
            for parameter in self.vqvae.vq_layer.parameters():
                parameter.requires_grad = False

        self.loss = None

    def load_checkpoint(self, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        vqvae_checkpoint = OrderedDict()

        for key in checkpoint.keys():
            if key.startswith("vqvae."):
                new_key = key.split(".")
                new_key = ".".join(new_key[1:])
                vqvae_checkpoint[new_key] = checkpoint[key]

        vqvae_ckpt_load_result = self.vqvae.load_state_dict(vqvae_checkpoint, strict=True)
        print(f"Load vae checkpoint from {checkpoint_path}, load result: {vqvae_ckpt_load_result}")

    def forward(self, act, robot_type=None, frequency=None):
        commit_loss, recon_loss, loss = self.vqvae(act, robot_type=robot_type, frequency=frequency, return_dict=False)
        return commit_loss, recon_loss, loss

    def get_code(self, x):
        with torch.no_grad():
            latents = self.vqvae.encode(x).latents
            B = latents.shape[0]  # B
            state_rep_flat = latents.view(latents.size(0), -1, latents.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vqvae.vq_layer(state_rep_flat)
            vq_code = vq_code.view(B, -1)  # b,2
        return vq_code

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vqvae.vq_layer.get_output_from_indices(encoding_indices)
        return z_embed

    def get_action_from_latent(self, latent, robot_type=None, control_frequency=None):
        dec = self.vqvae.decode(latent, robot_type, control_frequency)
        return dec

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
