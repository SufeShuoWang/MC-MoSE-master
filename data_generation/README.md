# Data Construction

## 1. Data Sources

Only processing code is provided here. Download the required inputs from their
providers and export the fields expected by `config/schemas.yaml`. The filenames
below are local export names used by this project, not provider download filenames.
Place the inputs under `Data_processing/manual_processing/`, relative to this directory.

| Provider and product | Expected local files |
| --- | --- |
| [WRDS: CRSP Mutual Funds](https://wrds-www.wharton.upenn.edu/pages/get-data/) | `monthly_returns.csv`, `monthly_tna.csv`, `fund_fees.csv`, `fund_hdr.csv`, `fund_hdr_hist.csv`, `fund_style.csv` |
| [WRDS: MFLINKS](https://wrds-www.wharton.upenn.edu/documents/1159/MFLinks.pdf) | `crsp_fundno_wficn.csv`, `fundno_wficn.csv` |
| WRDS: Thomson/Refinitiv S12 Mutual Fund Holdings | `s12type1.csv`, `s12type3.csv` |
| WRDS: CRSP Stock | `monthly_stock_file.csv`, `daily_stock_file.csv`, `mthadjustfactors.csv`, `distribution_Information.csv` |
| WRDS: Compustat and CRSP/Compustat Merged | `compfunda.csv`, `ccm_linktable.csv` |
| [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | `F-F_Research_Data_Factors.csv`, `F-F_Momentum_Factor.csv`, `F-F_Research_Data_Factors_daily.csv` |
| [Jeffrey Wurgler: Investor Sentiment](https://pages.stern.nyu.edu/~jwurgler/main.htm) | `SENTIMENT.xlsx` |
| [Chicago Fed: CFNAI](https://www.chicagofed.org/research/data/cfnai/current-data) | `CFNAI.csv` |

WRDS inputs require access to the corresponding products. Use `config/paths.yaml`
for paths, `config/schemas.yaml` for column requirements, and the configured
Sentiment worksheet (`DATA`). Save the CFNAI input in the configured CSV layout.
The original pipeline uses the historical coverage specified in its configuration.

## 2. Processing Pipeline

The numbered notebooks contain the processing steps and import their numerical,
configuration, and I/O functions from `src/`.

| Step | Processing |
| --- | --- |
| 00 | Load configuration and check inputs. |
| 01 | Inspect raw schemas, coverage, and keys. |
| 02 | Clean fund identifier links. |
| 03 | Build the share-class monthly panel. |
| 04 | Apply the original research-universe rules. |
| 05 | Aggregate share classes and fund characteristics. |
| 06 | Compute the alpha target and fund momentum. |
| 07 | Build management-company mappings. |
| 08 | Compute family characteristics. |
| 09 | Clean S12 report and holdings tables. |
| 10 | Map holdings fund identifiers to WFICN. |
| 11 | Map CUSIP to PERMNO. |
| 12 | Adjust shares and build report-date holdings. |
| 13 | Expand valid holdings to monthly records. |
| 14 | Prepare monthly/daily CRSP inputs and factor tables. |
| 15 | Prepare Compustat and CCM inputs. |
| 16 | Compute stock return characteristics. |
| 17 | Compute stock accounting characteristics. |
| 18 | Compute risk, liquidity, and out-of-sample SUV. |
| 19 | Combine the 46 stock characteristics. |
| 20 | Rank stock characteristics and apply PCA imputation. |
| 21 | Aggregate stock characteristics using fund holdings. |
| 22 | Rank fund and family characteristics. |
| 23 | Build the same-month characteristic panel. |
| 24 | Align predictors with target months. |
| 25 | Export production-format arrays. |

The first layout script converts the production arrays into the four-key model
layout. The second removes the CF characteristic while retaining all other values
and their order. Historical version-name assertions in notebook 00 have been
removed from this copy; computational checks are retained.

## 3. Output Summary

Notebook 25 exports `X`, `X_macro`, `y`, `dates`, and their index/name arrays.
The layout conversion produces `data`, `date`, `wficn`, and `names`, initially
with 62 channels. Removing CF produces the final 61-channel layout:

| Array | Shape | Meaning |
| --- | --- | --- |
| `data` | `(T, N, 61)` | Alpha target, 58 fund characteristics, and two market features. |
| `date` | `(T,)` | Target months in YYYYMM format. |
| `wficn` | `(N,)` | Fund identifiers. |
| `names` | `(60,)` | Names of predictor channels 1 through 60. |

Channel 0 is the alpha target. The market channels are `sentiment` and `CFNAI`.
Missing values use `-99.99`. Core characteristics and Sentiment are aligned to
target month t-1; CFNAI is aligned to t-2.

## 4. Commands

Run the commands from this directory. These document the existing processing
entry points; original inputs, reference layouts, and historical archive artifacts
used by the original workflow are not distributed.

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name fund-data --display-name "Fund data"
```

Execute an individual stage:

```bash
python -m src.execute_notebook notebooks/00_environment_and_config.ipynb --run-mode full --timeout 21600 --kernel-name fund-data
```

Execute stages 00 through 25 in filename order:

```bash
python -c "from pathlib import Path; from src.execute_notebook import execute_notebook; [execute_notebook(p, run_mode='full', timeout=21600, kernel_name='fund-data') for p in sorted(Path('notebooks').glob('*.ipynb')) if p.name[:2].isdigit() and 0 <= int(p.name[:2]) <= 25]"
```

Run the original final layout entry points:

```bash
python scripts/export_compatible_layout_npz.py --input INPUT.npz --reference REFERENCE_LAYOUT.npz --output COMPATIBLE.npz
python scripts/export_no_cf_compatible_layout_npz.py --input COMPATIBLE.npz --output FINAL.npz
```

Replace the uppercase paths with the corresponding local files. `--reference`
is an existing layout-script argument: it reads the original reference NPZ's
feature order, names, and dtypes. The reference file is not a model-training input
and is not included here. The original script defaults point to the original
versioned processing paths; explicit paths select the desired local artifacts.
