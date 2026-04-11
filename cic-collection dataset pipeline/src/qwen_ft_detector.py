from __future__ import annotations

import gc
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import canonicalize_type
from .utils import DEVICE, track


class QwenFTDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_prompt = self.build_system_prompt()
        self.tokenizer, self.model = self.load_qwen_ft()

    @staticmethod
    def build_system_prompt() -> str:
        return """You are a STRICT JSON-only CIC-Collection Zero-Day Attack Detector.

## IMPORTANT (READ CAREFULLY)
- You are NOT allowed to guess attack type from numeric patterns.
- You MUST be deterministic and consistent.
- You MUST use the provided \"label\" field when it exists.
- You MUST use the provided \"type\" field for zero-day decision when it exists.
- If \"type\" is missing/unknown, you MUST output zero_day_attack = \"no\".

## TASK
For each network-flow record in the input JSON array:
1) Predict whether the record is \"Benign\" or \"Attack\".
2) Decide whether it is a \"zero_day_attack\" (\"yes\"/\"no\") based ONLY on the rules below.

## ZERO-DAY SETUP (fixed)
- Normal label is: 0  (Benign)
- Attack label is: 1  (Attack)
- Attack category field name is: \"type\"

- ZERO_DAY_ATTACK_TYPES = [
    \"Webattack\",
  ]

- KNOWN_ATTACK_TYPES = [
    \"Portscan\",
    \"DDoS\",
    \"DoS\",
    \"Botnet\"
     \"Bruteforce\",
    \"Infiltration\" 
  ]

## STRICT DECISION LOGIC (MUST FOLLOW EXACTLY)

For each row:

Step A: Determine predicted_value
1) If \"label\" exists:
   - If label == 0 => predicted_value = \"Benign\"
   - If label == 1 => predicted_value = \"Attack\"
   - If label is any other value or not an integer => treat as missing and go to step A2
2) Else if \"type\" exists and type is not \"-\" / \"\" / null:
   - predicted_value = \"Attack\"
3) Else:
   - predicted_value = \"Benign\" (default; do not guess)

Step B: Determine zero_day_attack
1) If predicted_value == \"Benign\":
   => zero_day_attack = \"no\"
2) If predicted_value == \"Attack\":
   - If \"type\" exists and is not \"-\" / \"\" / null:
       - If type in ZERO_DAY_ATTACK_TYPES => zero_day_attack = \"yes\"
       - Else => zero_day_attack = \"no\"
   - Else:
       => zero_day_attack = \"no\"

## INPUT FORMAT
- You will receive a JSON array of rows.
- Every row may have an \"id\" field. If present, you MUST copy it into output. If missing, output id = null.
- Treat \"-\" as unknown/missing.
- Do NOT ask questions; always produce an output.

## OUTPUT FORMAT (STRICT)
Return ONLY a valid JSON array.
Each output item corresponds to the same-index input record.
No extra text. No markdown. No explanations.

## OUTPUT SCHEMA
[
  {
    \"id\": <copied from input or null>,
    \"predicted_value\": \"Benign\" | \"Attack\",
    \"zero_day_attack\": \"yes\" | \"no\"
  }
]

## ABSOLUTE CONSTRAINTS
- Output ONLY JSON array, nothing else.
- Do not output any keys other than: id, predicted_value, zero_day_attack.
- Do not output probabilities.
- Do not output explanations.
- Do not hallucinate attack types or labels.
- If \"type\" is missing/unknown, zero_day_attack MUST be \"no\".
- If missing info for predicted_value, default predicted_value to \"Benign\".
"""

    @staticmethod
    def safe_json_array_parse(text: Optional[str]):
        if text is None:
            return None
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except Exception:
                pass
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _safe_int_or_none(value: object) -> Optional[int]:
        try:
            if pd.isna(value):
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_json_value(value: object) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    def make_single_row_record(self, row: pd.Series, row_id: int, extra_cols: List[str]) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"id": int(row_id) if row_id is not None else None}
        if "label" in row:
            rec["label"] = self._safe_int_or_none(row["label"])
        elif "is_attack" in row:
            rec["label"] = self._safe_int_or_none(row["is_attack"])
        else:
            rec["label"] = None

        if "type" in row:
            rec["type"] = canonicalize_type(row["type"])
        elif "ClassLabel" in row:
            rec["type"] = canonicalize_type(row["ClassLabel"])
        elif "Label" in row:
            rec["type"] = canonicalize_type(row["Label"])
        else:
            rec["type"] = "Other"

        for col in extra_cols:
            if col in row:
                rec[col] = self._safe_json_value(row[col])
        return rec

    def load_qwen_ft(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        track("Loading Qwen base + LoRA adapter...")
        from peft import PeftModel

        model_name = self.config["base_model_name"]
        adapter_repo = self.config["hf_repo_name"]

        tok = AutoTokenizer.from_pretrained(adapter_repo, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
            track("Loaded base model in 4-bit.")
        except Exception as exc:
            track(f"4-bit load failed, falling back to fp16. Reason: {type(exc).__name__}: {exc}")
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(base, adapter_repo)
        model.eval()
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
        return tok, model

    def predict_attack_proba(
        self,
        df_raw_eval: pd.DataFrame,
        feature_cols: List[str],
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = batch_size or self.config.get("qwen_batch_size", 64)
        raw_cols = [
            c for c in df_raw_eval.columns
            if c not in ["label", "type", "type_norm", "is_attack", "is_zero_day", "Label", "ClassLabel"]
        ]
        prompt_extra_cols = raw_cols[: self.config.get("prompt_feature_limit", 12)]

        attack_proba: List[float] = []
        zero_day_pred: List[int] = []
        prompts = []
        for i in range(len(df_raw_eval)):
            row = df_raw_eval.iloc[i]
            rec = self.make_single_row_record(row, row_id=i, extra_cols=prompt_extra_cols)
            user_content = json.dumps([rec], ensure_ascii=False)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        for start in tqdm(range(0, len(prompts), batch_size), desc="QwenFT inference"):
            batch_prompts = prompts[start : start + batch_size]
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=self.config.get("max_new_tokens", 128),
                )

            input_lengths = enc["attention_mask"].sum(dim=1)
            for b in range(out.shape[0]):
                gen_ids = out[b, input_lengths[b]:]
                txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                arr = self.safe_json_array_parse(txt)
                if not arr or not isinstance(arr, list) or len(arr) < 1:
                    attack_proba.append(0.01)
                    zero_day_pred.append(0)
                    continue

                item = arr[0] if isinstance(arr[0], dict) else {}
                pv = str(item.get("predicted_value", "Benign")).strip().lower()
                zd = str(item.get("zero_day_attack", "no")).strip().lower()

                attack_proba.append(0.99 if pv == "attack" else 0.01)
                zero_day_pred.append(1 if zd == "yes" else 0)

        return np.array(attack_proba, dtype=np.float32), np.array(zero_day_pred, dtype=np.int32)

    def cleanup(self) -> None:
        del self.model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
