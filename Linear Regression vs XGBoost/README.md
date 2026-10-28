# Bike Sharing Demand: Linear Regression vs XGBoost

Predicting hourly bike rentals with a linear baseline and a gradient boosting model, and showing why tree ensembles win on non-linear tabular data.

| Model | Test MAE | Test RMSE | Test R² |
| --- | --- | --- | --- |
| Linear Regression | 74.11 | 100.45 | 0.681 |
| **XGBoost** | **23.03** | **37.83** | **0.955** |

XGBoost cuts the average error by about 69%.

## Dataset

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), file `hour.csv`: 17,379 hourly records from Capital Bikeshare, Washington D.C. (2011–2012).

- **Target:** `cnt`, total rentals in that hour
- **Features:** season, year, month, hour, holiday, weekday, working day, weather, temperature, feels-like temperature, humidity, wind speed
- **Dropped:** `casual` and `registered` (they sum to `cnt`, so keeping them leaks the target), `instant`, `dteday`

The dataset is not included in this repository. Download it from the link above and place `hour.csv` in the project root.

## Why this dataset

Rentals follow a strong non-linear daily pattern that changes between working days and weekends: two commuter peaks (08:00 and 17:00–18:00) on working days, one midday curve on weekends.
A linear model learns one fixed effect per hour, the same for Monday and Sunday. It cannot express this interaction. XGBoost learns it from tree splits without manual feature engineering.

## Approach

Both notebooks use the same 80/20 train-test split (`random_state=42`).

**Linear Regression** (`LinReg_BikeShare.ipynb`)
- scikit-learn `Pipeline` with `ColumnTransformer`
- One-hot encoding for categorical columns stored as numbers (`season`, `mnth`, `hr`, `weekday`, `weathersit`)
- Standard scaling for numeric columns
- Coefficient analysis, including the learned effect of each hour

**XGBoost** (`XGB_BikeShare.ipynb`)
- Raw features, no scaling or encoding needed
- 15% of the training set held out as validation for early stopping
- `learning_rate=0.05`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`, `reg_lambda=1.0`, `tree_method="hist"`
- Early stopping after 100 rounds without improvement: best iteration 573 (574 trees)
- Learning curve, first-tree structure, gain-based feature importance
- Step-by-step boosting: the same prediction built from the first 1, 5, 20, … 574 trees

## Results

| Model | Split | MAE | RMSE | R² |
| --- | --- | --- | --- | --- |
| Linear Regression | Train | 75.56 | 101.92 | 0.687 |
| Linear Regression | Test | 74.11 | 100.45 | 0.681 |
| XGBoost | Train | 14.20 | 23.21 | 0.984 |
| XGBoost | Test | 23.03 | 37.83 | 0.955 |

Linear Regression scores the same on train and test: it underfits. XGBoost generalizes well, with early stopping keeping the train-test gap small.

**Scenario test** on hand-made inputs neither model saw:

| Scenario | Linear Regression | XGBoost |
| --- | --- | --- |
| Working day 08:00, sunny, 20°C | 447.1 | 653.2 |
| Working day 17:00, sunny, 20°C | 513.6 | 629.6 |
| Sunday 08:00, sunny, 20°C | 434.9 | 161.7 |
| Working day 03:00, rainy, 8°C | −134.2 | −9.4 |

Linear Regression predicts nearly the same for a working-day morning and a Sunday morning. XGBoost captures the difference.

Top features by gain (XGBoost): `hr` 28.6%, `workingday` 22.7%, `yr` 18.4%.

## Project structure

```
.
├── LinReg_BikeShare.ipynb
├── XGB_BikeShare.ipynb
├── graphs/
│   ├── hourly_sharing_table.png
│   ├── lr_hour_coefs.png
│   ├── lr_results.png
│   ├── xgb_learning_curve.png
│   ├── xgb_feature_importance.png
│   └── xgb_results.png
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Download `hour.csv`, place it in the project root, set `DATA_PATH = "hour.csv"` in the first cell of each notebook, then run the notebooks top to bottom. Figures are saved to `graphs/`.

`requirements.txt`:

```
pandas
numpy
scikit-learn
xgboost
matplotlib
```

## Limitations and next steps

- Both models produce some negative predictions (319 of 3,476 test rows for Linear Regression, 75 for XGBoost). Training on `log(cnt)` would fix this.
- Linear Regression could close part of the gap with interaction features such as `hr × workingday`.
- The split is random. A time-based split (train on 2011, test on 2012) would better reflect real forecasting.

## Article

A detailed write-up covering gradient boosting theory, XGBoost internals and this comparison: *[Medium link]*

