"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        # breakpoint()
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids

        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        # print(discretized_actions)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


class VQVAEActionTokenizer:
    def __init__(
        self,
        vqvae_model,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        offset: int = 256,
    ) -> None:
        """
        VQVAE-based action tokenizer that encodes continuous robot actions into discrete token IDs.

        This tokenizer uses a pre-trained VQVAE model to compress continuous actions into discrete
        codebook indices, then maps them to the end portion of a pre-trained language model's
        vocabulary to achieve tokenized action representation.

        Args:
            vqvae_model: Pre-trained VQVAE model for action encoding/decoding
            tokenizer: Base LLM tokenizer providing the vocabulary
            bins: Codebook size for each VQVAE group, default 256
            min_action: Minimum action value range (for normalization), default -1
            max_action: Maximum action value range (for normalization), default 1
            offset: Offset between different VQVAE groups, default 256
        """
        self.vqvae_model, self.tokenizer, self.n_bins, self.min_action, self.max_action = (
            vqvae_model,
            tokenizer,
            bins,
            min_action,
            max_action,
        )
        self.offset = offset
        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(
            self.tokenizer.vocab_size - (self.offset * self.vqvae_model.vqvae.vqvae_groups + 1)
        )

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        discretized_action = self.vqvae_model.get_code(action)  # 1,4
        offsets = torch.tensor([0, self.offset * 1, self.offset * 2, self.offset * 3]).to(discretized_action.device)
        discretized_action = discretized_action + offsets
        # print(discretized_action)
        if len(discretized_action.shape) > 1:
            discretized_action = discretized_action.squeeze(0).cpu().numpy()
        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action - 1))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action - 1).tolist())

    def decode_token_ids_to_actions(
        self, action_token_ids: np.ndarray, robot_type=None, control_frequency=None
    ) -> np.ndarray:
        discretized_actions = self.tokenizer.vocab_size - action_token_ids - 1

        offsets = np.array([0, -self.offset * 1, -self.offset * 2, -self.offset * 3])
        discretized_actions = discretized_actions.reshape(-1, self.vqvae_model.token_num)
        discretized_actions = discretized_actions + offsets
        discretized_actions = np.clip(discretized_actions, a_min=0, a_max=self.offset - 1)
        discretized_actions = torch.from_numpy(discretized_actions).to(self.vqvae_model.device)
        output = self.vqvae_model.draw_code_forward(discretized_actions)  # b,laten_dim
        output = self.vqvae_model.get_action_from_latent(output, robot_type, control_frequency)
        return output

    @property
    def vocab_size(self) -> int:
        return self.n_bins
