from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "files": {
        "train": "UNSW_NB15_training-set.csv",
        "test": "UNSW_NB15_testing-set.csv",
    },
    "max_rows_total": 250000,
    "random_state": 42,
    "n_features": 35,
    "zero_day_attack_types": ["Analysis", "Shellcode", "Worms"],
    "gpt2_sequence_length": 32,
    "gpt2_hidden_size": 512,
    "gpt2_num_layers": 4,
    "gpt2_num_heads": 8,
    "gpt2_dropout": 0.1,
    "eo_population_size": 30,
    "eo_iterations": 50,
    "eo_candidate_count": 4,
    "sbd_percentile": 95,
    "sbd_min_violations": 1,
    "abd_n_components": 5,
    "abd_k": 2,
    "abd_epsilon": 1.6,
    # Fine-tuned Qwen3 branch
    "enable_qwenft": True,
    "hf_repo_name": " ", # provide necessary hugging face repository name after fine tuning
    "llm_val_max": 500,
    "llm_test_max": 4000,
    "qwen_batch_size": 64,
    "prompt_feature_limit": 12,
    "max_prompt_tokens": 384,
    "max_new_tokens": 96,
    "qwen_include_ground_truth_fields": False,
}


def get_config() -> Dict[str, Any]:
    return deepcopy(DEFAULT_CONFIG)
