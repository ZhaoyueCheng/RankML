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

import ml_collections

def get_config_dlrm_v2():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.001
    config.batch_size = 128
    config.num_epochs = 10
    config.num_dense_features = 13
    config.num_embedding_features = 26
    config.vocab_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                          11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
                          21000, 22000, 23000, 24000, 25000, 26000]  # 26 different vocab sizes
    config.embedding_dim = 32
    config.bottom_mlp_dims = (64, 32, 16)
    config.top_mlp_dims = (64, 32, 16)
    return config