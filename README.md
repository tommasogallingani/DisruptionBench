# DisruptionBench: Benchmarking for Disruption Prediction Models

Welcome to **DisruptionBench**, a comprehensive benchmarking class specifically designed for evaluating machine learning-driven disruption prediction models in tokamak reactors. This repository is created to facilitate the assessment of your models using a robust benchmarking framework, as detailed in our pre-print, "DisruptionBench: A robust benchmarking framework for machine learning-driven disruption prediction."

## Getting Started

To begin using DisruptionBench, please explore the `model_performance.ipynb` notebook provided in this repository. This notebook serves as a practical guide on how to utilize the DisruptionBench class with your disruption prediction models. You can directly copy the notebook and input your own data files for evaluation.

## How to install
You can download the latest version of the repository from the GitHub page and then you can install the package with pip:

```bash
pip install disruptionbench
```

## Install editable from source

You can install the package with:

```bash
pip install -e .
```


### Prerequisites

Before diving into the benchmarking process, ensure your data aligns with the expected input dimensions. You might need to use `np.unsqueeze(-1)` or a reshape command to add an extra dimension at the end of your data arrays, matching the DisruptionBench input requirements.

### Benchmarking Process

The core function of DisruptionBench is to compute a metrics report for your model's performance. Here is a snippet on how to invoke this computation:

```python
from disruptionbench.metrics import ModelEvaluator
import numpy as np
# Define your model parameters
params_dict = {
    'high_thr':.5, # high threshold [-]
    'low_thr':.5, # low threshold [-]
    't_hysteresis':0, # number of consecutive seconds above the high threshold required before triggering an alarm [s]
    't_useful':.005 # time before the end of the shot during which the disruption warning system's alerts are useful [s]
    }

metrics  = [
    'f1_score', 
    'f2_score', 
    'recall_score', 
    'precision_score', 
    'roc_auc_score', 
    'accuracy_score', 
    'confusion_matrix', 
    'tpr', 
    'fpr', 
    'AUC_zhu',
    'Warning_curve',
]

# Suppose your model predictions are stored in a dictionary named prediction_results
prediction_results = {
    'shot_1': {
        'proba_shot': np.array([0.1, 0.2, 0.3, 0.4, 0.5]), # model prediction
        'time_until_disrupt': np.array([0.1, 0.2, 0.3, 0.4, 0.5]), # time until disruption 
        'time_shot': np.array([0.005, 0.01, 0.015, 0.02, 0.025]), # time shot
        'label_shot': 1 # 0 for no disruption, 1 for disruption, true label
    },
    'shot_2': {
        'proba_shot': np.array([0.6, 0.7, 0.8, 0.9, 1.0]), # model prediction
        'time_until_disrupt': np.array([0.1, 0.2, 0.3, 0.4, 0.5]), # time until disruption 
        'time_shot': np.array([0.005, 0.01, 0.015, 0.02, 0.025]), # time shot
        'label_shot': 1 # 0 for no disruption, 1 for disruption, true label
    }
}
modeleval  = ModelEvaluator()
metrics = modeleval.eval(
    unrolled_proba = prediction_results,
    metrics = metrics,
    params_dict = params_dict
)

```

To use this functionality, you must prepare your model's disruptivity scores in a dictionary named `metrics`, adhering to the structure described in our documentation.

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

The list of available metrics is the following:

- `f1_score`: The F1 score, a harmonic mean of precision and recall.
- `f2_score`: The F2 score, emphasizing recall over precision.
- `recall_score`: The recall score, indicating the proportion of true positives identified.
- `precision_score`: The precision score, indicating the proportion of true positives among all positive predictions.
- `roc_auc_score`: The area under the ROC curve, a measure of the model's ability to distinguish between classes.
- `accuracy_score`: The overall accuracy of the model.
- `confusion_matrix`: A matrix showing the counts of true positives, false positives, true negatives, and false negatives.
- `tpr`: The true positive rate, also known as sensitivity or recall.
- `fpr`: The false positive rate, indicating the proportion of negatives incorrectly classified as positives.
- `tpr_at_5%fpr_zhu`: The true positive rate at a 5% false positive rate, a specific metric for evaluating disruption prediction models.
- `AUC_zhu`: The Zhu AUC, a specific metric for evaluating disruption prediction models as done in Zhu et al. 2020.
- `Warning_curve`: A curve showing the relationship between the number of warnings and the time until disruption.

## Citation

If you utilize DisruptionBench in your research, please cite our work as follows:

```
Spangher, et al. "DisruptionBench and Complimentary New Models: Two Advancements in Machine Learning Driven Disruption Prediction." 2024. DOI: 10.1007/s10894-025-00495-2.
```

Thank you for considering DisruptionBench for evaluating your disruption prediction models. 
We hope this tool aids in advancing the field of tokamak plasma stability research.
