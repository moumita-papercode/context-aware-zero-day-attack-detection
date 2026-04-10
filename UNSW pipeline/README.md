# UNSW-NB15 Zero-Day IDS Pipeline




## Structure

```text
unsw_zero_day_repo_updated/
├── run_unsw_pipeline.py
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
    ├── fusion.py
    └── evaluation.py
```

## File mapping from the notebook

- `config.py` -> notebook config cell
- `data.py` -> loading, zero-day marking, split, preprocessing
- `feature_selection.py` -> EO-based feature selection
- `traditional_detectors.py` -> SBD and ABD
- `neural_detectors.py` -> DNN and GPT-2
- `fusion.py` -> hybrid fusion logic
- `evaluation.py` -> threshold search and metrics
- `run_unsw_pipeline.py` -> runs the full pipeline end-to-end

## Run

```bash
pip install -r requirements.txt
python run_unsw_pipeline.py
```

## Notes

- Update `src/config.py` if your paths differ.

