import json
from disruptionbench.metrics import ModelEvaluator
with open('assets/example_val_shots_res.json', 'rb') as f:
    dict_y_proba_shots_val = json.load(f)


with open('assets/example_test_shots_res.json', 'rb') as f:
    dict_y_proba_shots_test = json.load(f)



# Necessary inputs
params_dict = {
    'high_thr':.8,
    'low_thr':.05,
    't_hysteresis':0.0,
    't_useful':.05
    }

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
    'Warning_curve']



m  = ModelEvaluator()
r = m.eval(
    unrolled_proba = dict_y_proba_shots_val,
    metrics = metrics,
    params_dict = params_dict
)




