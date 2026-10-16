# MC-MoSE

## Installation

Use Python 3.10.

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

The script runs ten independent random seeds. To run a subset:

```bash
python train.py --seeds 1
python train.py --seeds 1 12 123
```
## Data

Due to data restrictions, we provide only demonstration datasets in MC-MoSE-master/data/, comprising fund data for 2023–2024 and data required for backtesting for 2021–2024. The full training dataset is not included.

## Backtesting

```bash
python backtest.py
```

To evaluate a prediction file or another prediction directory:

```bash
python backtest.py --predictions outputs/test_predictions/mc_mose_seed1_test_predictions.csv
```

Sortino is left blank when all monthly long-short returns in the evaluated period are
nonnegative. SR is left blank when their standard deviation is zero.

Summary statistics use the valid seeds for each metric; `n_seeds` reports
that count. The mean is blank when no seeds are valid, and the sample standard
deviation is blank when fewer than two seeds are valid.

## Outputs

Training results are saved under `outputs/checkpoints/`, `outputs/logs/`,
`outputs/test_predictions/`, and `outputs/summaries/`.

Backtesting writes only `outputs/backtest.xlsx`, containing four worksheets:
`Summary`, `PerSeedMetrics`, `CumulativeReturn`, and `CumulativeAbnormalReturn`.
The evaluated months follow the prediction files.
