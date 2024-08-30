# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DLRM V2 Recommendation example.

Library file which executes the training and evaluation loop for DLRM V2.
The data is generated as fake input data.
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
from models import DLRMV2
from configs import get_config_dlrm_v2
from losses import bce_with_logits_loss
from metrics import accuracy

@jax.jit
def apply_model(state, dense_features, embedding_ids, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, dense_features, embedding_ids)
        loss = bce_with_logits_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    acc = accuracy(logits, labels)
    return grads, loss, acc

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['dense_features'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_dense = train_ds['dense_features'][perm]
        batch_embedding_ids = train_ds['embedding_ids'][perm]
        batch_labels = train_ds['labels'][perm]
        grads, loss, accuracy = apply_model(state, batch_dense, batch_embedding_ids, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy

def get_fake_datasets(config):
    """Generate fake datasets for DLRM V2."""
    num_samples = 10000
    rng = np.random.default_rng(42)

    dense_features = rng.random((num_samples, config.num_dense_features))
    embedding_ids = np.stack([rng.integers(0, vocab_size, num_samples) for vocab_size in config.vocab_sizes], axis=1)
    labels = rng.integers(0, 2, (num_samples,))

    split = int(0.8 * num_samples)
    train_ds = {
        'dense_features': jnp.array(dense_features[:split]),
        'embedding_ids': jnp.array(embedding_ids[:split]),
        'labels': jnp.array(labels[:split])
    }
    test_ds = {
        'dense_features': jnp.array(dense_features[split:]),
        'embedding_ids': jnp.array(embedding_ids[split:]),
        'labels': jnp.array(labels[split:])
    }
    return train_ds, test_ds

def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    dlrm = DLRMV2(
        vocab_sizes=config.vocab_sizes,
        embedding_dim=config.embedding_dim,
        bottom_mlp_dims=config.bottom_mlp_dims,
        top_mlp_dims=config.top_mlp_dims
    )
    params = dlrm.init(
        rng, 
        jnp.ones([1, config.num_dense_features]), 
        jnp.ones([1, len(config.vocab_sizes)], dtype=jnp.int32)
    )['params']
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=dlrm.apply, params=params, tx=tx)

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop."""
    train_ds, test_ds = get_fake_datasets(config)
    rng = jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds, config.batch_size, input_rng)
        _, test_loss, test_accuracy = apply_model(
            state, test_ds['dense_features'], test_ds['embedding_ids'], test_ds['labels']
        )

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )
        print('epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100))

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test_loss, epoch)
        summary_writer.scalar('test_accuracy', test_accuracy, epoch)

    summary_writer.flush()
    return state

if __name__ == "__main__":
    train_and_evaluate(get_config_dlrm_v2(), '/tmp/dlrm_v2')
