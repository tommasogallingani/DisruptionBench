# DisruptionBench: Benchmarking for Disruption Prediction Models

Welcome to **DisruptionBench**, a comprehensive benchmarking class specifically designed for evaluating machine learning-driven disruption prediction models in tokamak reactors. This repository is created to facilitate the assessment of your models using a robust benchmarking framework, as detailed in our pre-print, "DisruptionBench: A robust benchmarking framework for machine learning-driven disruption prediction."

## Getting Started

To begin using DisruptionBench, please explore the `model_performance.ipynb` notebook provided in this repository. This notebook serves as a practical guide on how to utilize the DisruptionBench class with your disruption prediction models. You can directly copy the notebook and input your own data files for evaluation.

### Prerequisites

Before diving into the benchmarking process, ensure your data aligns with the expected input dimensions. You might need to use `np.unsqueeze(-1)` or a reshape command to add an extra dimension at the end of your data arrays, matching the DisruptionBench input requirements.

### Benchmarking Process

The core function of DisruptionBench is to compute a metrics report for your model's performance. Here is a snippet on how to invoke this computation:

```python
# Compute metrics_report
val_metrics_report = performance_calculator.eval(
    unrolled_proba = dict_y_proba_shots_val,
    metrics = metrics,
    params_dict = params_dict
)
```

To use this functionality, you must prepare your model's disruptivity scores in a dictionary named `dict_y_proba_shots_val`, adhering to the structure described in our documentation.

#### Important Notes:

- **Time until disruption:** You can calculate the "time until disruption" field by subtracting your time slices' indices from the sequence length and multiplying by the sampling frequency, as shown below:

    ```python
    (len(observation) - np.array(range(0, len(observation))) * sampling_frequency
    ```

- **Time shot:** This field should represent an incrementing count based on the sampling frequency. If your shots consistently use the same time base (e.g., 0.005 seconds), this column can simply contain incrementing ones.

### Configuration Parameters

The `params` dictionary within DisruptionBench allows you to specify various hyperparameters relevant to your model's evaluation. These parameters include:

- `high_thr`: The high threshold above which a disruption warning is considered valid.
- `low_thr`: The lower threshold that, when exceeded but below the high threshold, does not trigger a disruption warning; falling below this threshold resets the system.
- `t_hysteresis`: The number of consecutive time-steps above the high threshold required before triggering an alarm.
- `t_useful`: Defines the period prior to the end of a shot during which the disruption warning system's alerts are deemed useful. Alerts triggered beyond this period are not counted as valid positives.

## Citation

If you utilize DisruptionBench in your research, please cite our work as follows:

```
Spangher, et al. "DisruptionBench: A robust benchmarking framework for machine learning-driven disruption prediction." 2024. Pre-print.
```

Thank you for considering DisruptionBench for evaluating your disruption prediction models. We hope this tool aids in advancing the field of tokamak plasma stability research.
