# TON-IoT Zero-Day IDS Pipeline



## Structure

```text
ton_iot_zero_day_repo/
├── run_ton_iot_pipeline.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── data.py
    ├── feature_selection.py
    ├── traditional_detectors.py
    ├── neural_detectors.py
    ├── qwen_ft_detector.py
    ├── fusion.py
    └── evaluation.py
```

## File mapping from the notebooks

- `config.py` -> shared config, dataset path, Qwen base/adaptor settings, eval subset caps
- `data.py` -> TON-IoT CSV loading, zero-day marking, split, preprocessing
- `feature_selection.py` -> EO-based feature selection
- `traditional_detectors.py` -> SBD and ABD
- `neural_detectors.py` -> DNN and GPT-2
- `qwen_ft_detector.py` -> fine-tuned Qwen3 inference from raw TON-IoT rows
- `fusion.py` -> DNN/GPT-2 fusion 
- `evaluation.py` -> threshold search and metrics
- `run_ton_iot_pipeline.py` -> runs the full pipeline end-to-end

## Run

```bash
pip install -r requirements.txt
python run_ton_iot_pipeline.py
```

## Notes

- Update `src/config.py` if your CSV path or Qwen checkpoint names differ.
- `base_model_name` and `hf_repo_name` are placeholders; set them to your actual Qwen base model and LoRA adapter.
