# Electricity Demand Forecasting with XGBoost and LSTM

I compare a weekly seasonal baseline, XGBoost and a long short-term memory (LSTM) neural network for next-day electricity-demand forecasting. My aim is to investigate whether the neural model provides an accuracy benefit over a strong tree-based model.

**Main finding:** Both learned models reduced held-out mean absolute error by approximately 25% relative to the weekly baseline. The LSTM had the lowest validation error, but XGBoost had a slightly lower test error. This experiment does not demonstrate a test-performance advantage for the LSTM.

## Forecasting task

At midnight each day, I predict the next 24 hourly demand readings using the previous 168 hours and calendar information known at the forecast origin. I use customer `MT_001` from the UCI ElectricityLoadDiagrams20112014 dataset.

I average each set of four quarter-hour readings to obtain hourly mean demand in kilowatts (kW). The retained 2012–2014 series contains 26,304 hourly readings. Quarter-hour timestamps are shifted back by 15 minutes to label their preceding intervals before aggregation.

## Evaluation design

| Split | Forecast origins | Daily forecasts |
| --- | --- | ---: |
| Training | 8 January 2012–31 December 2013 | 724 |
| Validation | 1 January–30 June 2014 | 181 |
| Test | 1 July–31 December 2014 | 184 |

The first seven days supply the initial history window. Test evaluation covers 4,416 hourly predictions. The target windows do not overlap.

I use chronological splits and fit the standardisation parameters only on training-period readings. Model weights remain fixed throughout evaluation. Each new daily forecast can use demand already observed earlier in the validation or test period; it cannot use readings from its own future target window.

I select model settings and the preferred model using validation mean absolute error (MAE). All three pre-specified models are then evaluated on the test set. I do not replace the validation-selected model retrospectively using test performance.

## Models

- **Weekly baseline:** repeats demand from the same hours in the previous week.
- **XGBoost:** a pooled direct forecasting model with historical demand, cyclical calendar features, a weekday indicator and a forecast-horizon feature. The selected configuration uses 250 trees, maximum depth 5 and learning rate 0.05, chosen from two candidates by validation MAE.
- **LSTM:** a 32-unit sequence encoder followed by a dense prediction head that also receives future calendar features. It predicts all 24 hours together. Training uses Adam, gradient clipping and validation-based early stopping. The best checkpoint was epoch 18; training stopped after epoch 23.

Both learned models receive the same history length and known calendar information, represented differently. Predictions are clipped at zero because demand is non-negative.

## Results

| Model | Validation MAE (kW) | Test MAE (kW) | Test RMSE (kW) | Test WAPE | Test MAE reduction vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weekly baseline | 2.0889 | 2.2345 | 4.8496 | 42.49% | — |
| XGBoost | 1.6101 | **1.6755** | **2.9746** | **31.86%** | **25.02%** |
| LSTM | **1.5351** | 1.6840 | 3.0302 | 32.03% | 24.63% |

MAE is the mean absolute prediction error. Root mean squared error (RMSE) gives larger errors more weight. Weighted absolute percentage error (WAPE) is total absolute error divided by total actual demand. Lower values are better; WAPE is not a classification accuracy score.

The full-precision scores are in [`results/metrics.csv`](results/metrics.csv). The executed notebook contains the forecast plots, error breakdowns and training curve.

### Interpretation

The LSTM was selected by validation MAE (1.5351 kW versus 1.6101 kW for XGBoost). On the held-out test set, XGBoost's MAE was approximately 0.0086 kW lower than the LSTM's—about a 0.51% reduction relative to the LSTM. This small difference is descriptive; I have not established statistical significance.

The saved run used **CPU**, including a CPU build of PyTorch. Measured fitting and selection time was approximately 2.57 seconds for XGBoost and 7.81 seconds for the LSTM. These timings exclude data loading and are specific to this run. They cover different selection procedures and are not a controlled hardware benchmark.

The largest daily LSTM error in the displayed error table occurred on **1 August 2014**, with MAE of approximately 14.33 kW. All three models performed poorly on that day. The outputs identify a useful case for investigation, but do not establish the cause of the error.

## Repository files

Place the executed notebook at the repository root and the supplied metrics file in `results/`.

| Path | Contents |
| --- | --- |
| `README.md` | Project summary, methodology and findings |
| `electricity-demand-forecasting-can-deep-learning.ipynb` | Executed analysis, code, tables and plots |
| `results/metrics.csv` | Full-precision evaluation scores |

Running the notebook also writes `forecast_comparison.png`, `test_predictions.csv`, `lstm_training_history.csv`, `experiment_metadata.json`, `lstm_checkpoint.pt` and `xgboost_model.json` to `electricity_results`. These additional outputs are not required to read the executed notebook. Saved model weights do not include the original dataset.

## Running the notebook

### Kaggle

1. Import `electricity-demand-forecasting-can-deep-learning.ipynb` into a Kaggle notebook.
2. Keep `DEMO_MODE = False` to use the real dataset.
3. Enable Internet for automatic data downloading, or attach the original UCI dataset ZIP/TXT through Kaggle's input panel. The loader searches attached data for `LD2011_2014.txt` or the original ZIP filename.
4. For a differently named file or a specific input, set `LOCAL_DATA_PATH` to its exact path.
5. Run all cells from top to bottom. CPU is sufficient. A GPU is optional for the LSTM and requires a CUDA-enabled PyTorch installation; the setup cell reports the detected device.
6. Find generated outputs in `/kaggle/working/electricity_results`.

The original ZIP download is approximately 249 MB. A DNS/download error can be avoided by downloading it separately and attaching it as input. The notebook does not silently substitute synthetic data when downloading fails.

### Local Jupyter

Use an environment with NumPy, pandas, Matplotlib, scikit-learn, XGBoost, PyTorch, requests and IPython, plus Jupyter to open the notebook. The notebook installs missing core packages when Internet access is available. Set `LOCAL_DATA_PATH` to the downloaded ZIP or extracted TXT file, then run all cells. Outside Kaggle, outputs are written under the current working directory.

### Versions recorded in the submitted run

| Package | Version |
| --- | --- |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.2.0 |
| PyTorch | 2.10.0+cpu |

The random seed is 42. The notebook also uses deterministic cuDNN settings when applicable. Results can vary with package versions, hardware and numerical implementations. The recorded versions above are not a complete environment lockfile.

## Limitations

- One customer, one validation block and six months of test forecasts limit generalisability.
- The comparison uses a small tree search and a single LSTM architecture and seed.
- The source uses a local-time convention with daylight-saving artefacts, including artificial zeros and aggregated readings. These are retained.
- Weather, holidays and other external factors are not model inputs.
- Prediction intervals, repeated-seed estimates and statistical significance tests are not included.
- The notebook's conclusion still requires updating from its original placeholders using the completed findings reported here.

## Further work

I plan to evaluate additional customers, use expanding-window validation and repeat neural training with multiple seeds. I would test the contribution of calendar features and different history lengths, investigate unusual demand periods and add calibrated prediction intervals. Further model development informed by the current test results would require a fresh held-out evaluation period.

## Data source and attribution

Trindade, A. (2015). *ElectricityLoadDiagrams20112014* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C58C86](https://doi.org/10.24432/C58C86).

[Dataset page](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)

The dataset is distributed under CC BY 4.0. The original dataset is obtained separately and is not bundled in this repository. The dataset licence does not by itself specify a licence for the project code.
