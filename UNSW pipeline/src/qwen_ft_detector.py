from __future__ import annotations

import gc
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import DEVICE


SYSTEM_PROMPT_SHORT = """You are a STRICT JSON-only UNSW-NB15 Zero-Day Attack Detector.
Input: a JSON array of rows. Output: a JSON array of same length.
Each output item MUST be: {"id":<copied>,"predicted_value":"Normal"|"Attack","zero_day_attack":"yes"|"no"} and NOTHING else.

Rules:
predicted_value:
- if label exists: label==0 -> Normal, label==1 -> Attack, else treat missing
- else if attack_cat exists and not "-" "" null: attack_cat=="Normal" -> Normal, else -> Attack
- else predicted_value="Normal"
zero_day_attack:
- if predicted_value=="Normal" -> "no"
- if predicted_value=="Attack" and attack_cat in ["Analysis","Shellcode","Worms"] -> "yes"
- otherwise -> "no"
If attack_cat missing/unknown/"-" => zero_day_attack MUST be "no".
Return ONLY JSON array, no extra keys/text.
""".strip()

EXCLUDE_FOR_QWEN = {
    "is_attack",
    "is_benign",
    "attack_type",
    "is_zero_day",
    "split",
    "label_binary",
}


class QwenFTDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["hf_repo_name"],
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["hf_repo_name"],
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(DEVICE)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def row_to_input_dict(self, row: pd.Series) -> Dict[str, Any]:
        keys = [k for k in row.index if k not in EXCLUDE_FOR_QWEN]
        if "id" in keys:
            keys = ["id"] + [k for k in keys if k != "id"]

        prompt_feature_limit = self.config.get("prompt_feature_limit")
        if prompt_feature_limit is not None:
            keys = keys[: (1 + prompt_feature_limit)] if ("id" in keys) else keys[:prompt_feature_limit]

        payload: Dict[str, Any] = {}
        include_ground_truth = self.config.get("qwen_include_ground_truth_fields", False)
        for key in keys:
            if (not include_ground_truth) and key in {"label", "attack_cat"}:
                continue
            value = row[key]
            if pd.isna(value):
                continue
            payload[key] = value.item() if isinstance(value, np.generic) else value
        return payload

    @staticmethod
    def build_prompt_one_row(row_dict: Dict[str, Any]) -> str:
        user_json = json.dumps([row_dict], ensure_ascii=False)
        return SYSTEM_PROMPT_SHORT + "\n\n" + user_json + "\n"

    @staticmethod
    def safe_extract_json_array(text: str) -> Optional[list]:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

        left = text.find("[")
        right = text.rfind("]")
        if left != -1 and right != -1 and right > left:
            try:
                obj = json.loads(text[left : right + 1])
                if isinstance(obj, list):
                    return obj
            except Exception:
                return None
        return None

    @torch.inference_mode()
    def predict_proba(self, df_raw_eval: pd.DataFrame, batch_size: Optional[int] = None) -> np.ndarray:
        batch_size = batch_size or self.config.get("qwen_batch_size", 64)
        max_prompt_tokens = self.config.get("max_prompt_tokens", 384)
        max_new_tokens = self.config.get("max_new_tokens", 96)

        probs = []
        for i in tqdm(range(0, len(df_raw_eval), batch_size), desc="QwenFT generate"):
            chunk = df_raw_eval.iloc[i : i + batch_size]
            prompts = [self.build_prompt_one_row(self.row_to_input_dict(row)) for _, row in chunk.iterrows()]
            enc = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
                return_tensors="pt",
            ).to(DEVICE)

            input_lens = enc["attention_mask"].sum(dim=1)
            out_ids = self.model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            batch_p = []
            for j in range(out_ids.size(0)):
                gen_tok = out_ids[j, input_lens[j] :]
                gen_txt = self.tokenizer.decode(gen_tok, skip_special_tokens=True)
                arr = self.safe_extract_json_array(gen_txt)

                predicted_value = "Normal"
                if isinstance(arr, list) and len(arr) >= 1 and isinstance(arr[0], dict):
                    predicted_value = str(arr[0].get("predicted_value", "Normal"))
                batch_p.append(0.99 if predicted_value.lower() == "attack" else 0.01)

            probs.append(np.array(batch_p, dtype=np.float32))

            del enc, out_ids
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        all_probs = np.concatenate(probs, axis=0)
        return np.clip(all_probs, 0.01, 0.99)
