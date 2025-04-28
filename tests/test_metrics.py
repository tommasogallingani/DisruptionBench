from unittest import TestCase
from disruptionbench.metrics import ModelEvaluator
import json

path_asset = 'assets/example_test_shots_res.json'
metrics = [
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
    'Warning_curve'
]
PARAMS = {
    'high_thr':.5,
    'low_thr':.5,
    't_hysteresis':0,
    't_useful':.005
    }


class TestDummy(TestCase):
    def setUp(self):
        with open(path_asset, 'r') as f:
            self.data = json.load(f)

        self.model_eval = ModelEvaluator()


    def test_all_metrics(self):
        # Test all metrics
        results = self.model_eval.eval(
            unrolled_proba = self.data,
            metrics = metrics,
            params_dict = PARAMS
        )

        # Check that all metrics are present in the results
        for metric in metrics:
            self.assertIn(metric, results)

            # Check that the results are not empty
            self.assertIsNotNone(results[metric])






