# MATH840 final project: 48 hours of generation (Illia Narbut)

A forecast of the 48 hours after the end of `train.csv` (27 Oct 2026, 01:00 to 29 Oct 2026, 00:00), with a 95%
prediction interval, for the KSE MATH840 (Time Series) final project.

The report is the notebook **`Narbut_project.ipynb`**: it answers the four questions of the brief (Q1–Q4) and
produces the submission file **`NARBUT_project.csv`**.

## How to run

Python 3.10+ with:

```
pip install numpy pandas lightgbm scikit-learn matplotlib
```

Then, from this folder:

| what | command | time |
|---|---|---|
| the report and the submission file | open `Narbut_project.ipynb` and run all cells | about 2 min |
| only the submission file | `python make_forecast.py` | about 1 min |
| regenerate the 150-origin backtest (`backtest_final.csv`) | `python run_backtest.py` | 40–60 min on 8 cores |

The notebook and `make_forecast.py` read the backtest from `backtest_final.csv`, which is committed. If the file is
missing, or `RUN_BACKTEST = True` is set in the notebook, they recompute it first. Everything is seeded, so a
re-run gives the same file. Different LightGBM builds or thread counts can change the last digits.

## Files

| file | content |
|---|---|
| `Narbut_project.ipynb` | the report and the full pipeline (course template + my sections) |
| `forecast_lib.py` | features, the stacked LightGBM, the weather-noise model, the backtest, the interval and the MAPE adjustment |
| `run_backtest.py` | the 150-origin rolling backtest, with the Session 1 model as a reference |
| `make_forecast.py` | fits the final model on the whole history and writes `NARBUT_project.csv` |
| `backtest_final.csv` | backtest output: 150 origins × 48 hours (actual, seasonal naive, Session 1 model, new model) |
| `NARBUT_project.csv` | the final submission |
| `train.csv`, `weather_forecast.csv` | the data from Moodle |

## The approach in brief

* **One LightGBM for all 48 horizons.** Each training row is (target hour, horizon). The features describe the
  target hour (calendar, solar hour, raw/smoothed/day-mean weather) and the origin (recent levels, and the
  seasonal-naive value at the same solar hour).
* **Built for a growing portfolio.** Level features at the origin, and a second model of `y / 28-day level`. The
  point forecast is the mean of the two models.
* **Built for the clock change on 27 Oct.** The hour is measured from the middle of the daylight window, so the
  solar curve stays aligned.
* **Built for a noisy weather forecast.** The noise size is estimated from the roughness of
  `weather_forecast.csv`. It is used to simulate forecast weather in the backtest and to train the model on noisy
  weather.
* **95% interval.** Asymmetric quantile distances `a·vol + b·forecast` per daylight/dark × day group, fitted on
  backtest residuals. `vol` is the recent 14-day day-to-day volatility.
* **Tuned for the leaderboard rank.** The leaderboard ranks per metric; it does not average. I simulated it against
  12 competitor forecasts on 150 backtest days and chose two settings:
  * shrink forecasts below 500 (×0.6 near zero), because MAPE punishes over-forecasting near-zero calm nights;
  * make the interval 10% narrower than calibrated, because that wins about two days in three.

Backtest on 150 daily origins (29 May – 25 Oct 2026), with simulated forecast weather and leave-one-origin-out
calibration, on hours 25–48 (the ranked part):

| | RMSE | MAPE | Winkler | coverage | simulated rank (of 13) |
|---|---|---|---|---|---|
| seasonal naive (benchmark) | 272 | 225 | 1850 | 0.86 | |
| my Session 1 model + its interval | 224 | 145 | 1101 | 0.95 | |
| final model, calibrated interval | 191 | 89 | 916 | 0.95 | 4.33 |
| **final model, as submitted (interval ×0.9)** | **191** | **89** | **944** | **0.91** | **4.18** |

## Note on LLM use

<!-- Illia: please check this paragraph and edit it so that it describes your own use accurately. -->
I used Claude (Anthropic) as a coding and analysis assistant, in Session 1 and at home.

* **What I asked it to do:** write and refactor the pandas/LightGBM code, set up the rolling-origin backtest, run
  the experiments listed in Q4, and draft parts of the report text.
* **What I did myself:** I decided what to test and which results to trust. The key choices are mine: validating
  on day 2 with simulated forecast weather, the volatility-scaled interval, and the MAPE trade-off. I checked the
  numbers in the notebook against the code output.
* **What it did not use:** only the data provided on Moodle. No outside data, and no search for the original
  source of the series.
