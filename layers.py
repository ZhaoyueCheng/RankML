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

class MLP(nn.Module):
    """Multi-layer perceptron."""
    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.relu(x)
        return x

class DenseArch(nn.Module):
    """Dense features architecture."""
    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, dense_features):
        return MLP(self.layer_sizes)(dense_features)

class EmbeddingArch(nn.Module):
    """Embedding architecture."""
    num_embeddings: int
    embedding_dim: int

    @nn.compact
    def __call__(self, embedding_ids):
        embedding_table = self.param('embedding', nn.initializers.uniform(), (self.num_embeddings, self.embedding_dim))
        embeddings = jnp.take(embedding_table, embedding_ids, axis=0)
        return embeddings.reshape((embeddings.shape[0], -1))

class InteractionArch(nn.Module):
    """Interaction architecture."""

    @nn.compact
    def __call__(self, dense_output, embedding_output):
        return jnp.concatenate([dense_output, embedding_output], axis=1)

class OverArch(nn.Module):
    """Over-architecture (top MLP)."""
    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = MLP(self.layer_sizes)(x)
        return nn.Dense(features=1)(x)