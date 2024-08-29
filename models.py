"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Sequence
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from layers import MLP, DenseArch, EmbeddingArch, InteractionArch, OverArch

class DLRMV2(nn.Module):
    """DLRM V2 model."""
    num_embeddings: int
    embedding_dim: int
    bottom_mlp_dims: Sequence[int]
    top_mlp_dims: Sequence[int]

    def setup(self):
        self.dense_arch = DenseArch(self.bottom_mlp_dims)
        self.embedding_arch = EmbeddingArch(self.num_embeddings, self.embedding_dim)
        self.interaction_arch = InteractionArch()
        self.over_arch = OverArch(self.top_mlp_dims)

    def __call__(self, dense_features, embedding_ids):
        dense_output = self.dense_arch(dense_features)
        embedding_output = self.embedding_arch(embedding_ids)
        interaction_output = self.interaction_arch(dense_output, embedding_output)
        return self.over_arch(interaction_output).squeeze()