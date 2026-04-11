# CIC Collection Zero-Day IDS Pipeline


## Structure

```text
cic_collection_zero_day_repo/
├── run_cic_collection_pipeline.py
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

- `config.py` -> shared config, dataset path, Qwen base/adapter settings, eval subset caps
- `data.py` -> CIC Collection parquet loading, zero-day family marking, split, preprocessing
- `feature_selection.py` -> EO-based feature selection
- `traditional_detectors.py` -> SBD and ABD
- `neural_detectors.py` -> DNN and GPT-2
- `qwen_ft_detector.py` -> fine-tuned Qwen3 inference from raw CIC rows
- `fusion.py` -> DNN/GPT-2/Qwen fusion
- `evaluation.py` -> threshold search and metrics
- `run_cic_collection_pipeline.py` -> runs the full pipeline end-to-end

## Run

```bash
pip install -r requirements.txt
python run_cic_collection_pipeline.py
```

## Notes

- Update `src/config.py` if your parquet path or Qwen checkpoint names differ.
- `base_model_name` and `hf_repo_name` are placeholders; set them to your actual Qwen base model and LoRA adapter.

