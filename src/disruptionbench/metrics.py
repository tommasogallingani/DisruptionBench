import logging
from typing import List, Union

import numpy as np
from functools import partial
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger('Metrics')


class TooShortShot(Exception):
    """
    Exception raised when a shot is too short to be classified
    """
    pass


def fpr_score(
        y_true: np.ndarray,
        y_pred: np.ndarray
):
    """
    False positive rate

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: fpr

    """
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return fp / (fp + tn)


def tpr_score(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> float:
    """
    True positive rate

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: tpr

    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn)


def auc(
        x: np.ndarray,
        y: np.ndarray
) -> float:
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.


    :param x: x coordinates. These must be either monotonic increasing or monotonic decreasing.
    :param y: y coordinates.
    :return: auc

    """

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area


def compute_AUC_Zhu(
        unrolled_proba: dict,
        t_useful: float
) -> List[Union[float, np.ndarray]]:
    """
    Compute AUC using Zhu's routines

    :param unrolled_proba: dictionary of unrolled probabilities
    :param t_useful: useful time or mitigation time
    :return: auc_score, best_thre, fpr_s, tpr_s

    """
    Disruptivity = []
    testTime_until_disrupt = []
    testClasses = []
    for _, item in unrolled_proba.items():
        Disruptivity.append(item['proba_shot'])
        testTime_until_disrupt.append(item['time_until_disrupt'])
        testClasses.append(item['label_shot'])
    testClasses = np.array(testClasses)

    num_P = len(np.where(testClasses == 1)[0])  # number of disruptive shots in test dataset
    num_N = len(np.where(testClasses == 0)[0])  # number of non-disruptive shots in test dataset

    tpr_Zhu = []
    fpr_Zhu = []
    threshold_list = np.column_stack([
        np.linspace(0, .1, 1001).reshape(-1, 1),
        np.linspace(0.1, .9, 1001).reshape(-1, 1),
        np.linspace(.9, 1, 1001).reshape(-1, 1),
    ])
    threshold_list = np.unique(threshold_list)
    for i, t in enumerate(threshold_list):
        if i // 100 == i / 100:
            logger.info('AUC: loop over thresh. Done {} of {}'.format(i, len(threshold_list)))
        thresh = t
        num_test_sample = testClasses.shape[0]
        classes = np.zeros(num_test_sample, dtype='int16')  # predicted class of the shot, if disruptive or not
        for i in range(num_test_sample):
            index = np.where(Disruptivity[i] > thresh)
            if len(index[0]) > 0:
                classes[i] = 1  # if the predicted disruptivity > thresh -> label the shot as disruptive
                # ACHTUNG!!!: no check on where the disruptivity is > thresh, can happen at the beginning of the shot and then go to zero!!!

        indexx = np.where((classes == 1) & (testClasses == 1))[
            0]  # shot indexes -> where the shot class is predicted correctly
        testTime_until_disrupt_effec = [testTime_until_disrupt[i] for i in
                                        indexx]  # select time_until_disrupt for shots that are correctly labelled
        b = np.zeros(num_test_sample)
        for i in range(num_test_sample):  # loop over test shots
            index = np.where(Disruptivity[i] > thresh)  # check where disruptivity is > thresh
            if len(index[0]) > 0:  # if any instants where disruptivity > thresh
                b[i] = index[0][0]  # record the earliest time instant where disruptivity > thresh
        b = b[indexx]  # select time instants where the alarm is raised for truly disruptive shots
        b = np.asarray(b, dtype='int16')
        tin = b  # +10-1 # add 9 -> last time instant of the window
        time_new = []
        for i in range(len(tin)):  # loop over truly disruptive shots
            until = testTime_until_disrupt_effec[i].reshape(-1, 1).tolist()
            time_new.append(until[tin[i]])
        time_new = np.asarray(time_new)
        tpr_Zhu.append(len(
            np.where(time_new > t_useful)[0]) / num_P)  # label as true positive alarms raised at least 50ms in advance
        fpr_Zhu.append(len(np.where((classes == 1) & (testClasses == 0))[0]) / num_N)
    fpr = np.array(fpr_Zhu)
    tpr = np.array(tpr_Zhu)
    auc_score_Zhu = auc(fpr, tpr)
    auc_score = auc_score_Zhu
    fpr_s = np.array(fpr)[np.argsort(fpr)]
    tpr_s = np.array(tpr)[np.argsort(fpr)]

    # get best thr
    best_idx = np.argsort(-fpr_s + tpr_s + 1)[-1]

    best_thre = threshold_list[np.argsort(fpr)][best_idx]

    logger.info('AUC using Zhu\'s routines = {}'.format(auc_score))
    return auc_score, best_thre, fpr_s, tpr_s


def warning_curve(
        class_shots_pred,
        class_shots_true,
        time_alarm,
        time_until_disrupt_alarm
):
    """
    Compute warning curve

    :param class_shots_pred: class predicted
    :param class_shots_true: class true
    :param time_alarm: time alarm
    :param time_until_disrupt_alarm: time until disrupt when the alarm is raised
    :return: warning curve with x and y elements

    """
    # Build alarm curve
    tp_index = (class_shots_pred == class_shots_true) & (class_shots_true == 1)
    disrp_shots = np.sum(class_shots_true == 1)
    # Reorder and build curve
    tp_alarm_time = time_alarm[tp_index]
    tp_until_disrupt_alarm_time = time_until_disrupt_alarm[tp_index]
    tp_alarm_time_reorder = tp_alarm_time[np.argsort(-tp_alarm_time)]
    # Reorder and build curve
    tp_until_disrupt_alarm_time_reorder = tp_until_disrupt_alarm_time[np.argsort(-tp_until_disrupt_alarm_time)]
    cumulative_contribution = [x for x in range(tp_until_disrupt_alarm_time_reorder.shape[0])] / disrp_shots
    return [tp_until_disrupt_alarm_time_reorder, cumulative_contribution]


class ModelEvaluator:
    """
    Class for evaluating the model
    """

    def __init__(
            self
    ):

        self.time_until_disrupt_alarm = None
        self.metrics_report = None
        self.time_until_disrupt_alarm = None
        self.time_alarm_lst = None
        self.time_until_disrupt_alarm_lst = None
        self.class_shots_pred = None
        self.class_shots_true = None
        self.time_alarm = None
        self.time_until_disrupt_alarm = None

        self.metrics_dict = {
            'f1_score': f1_score,
            'recall_score': recall_score,
            'precision_score': precision_score,
            'balanced_accuracy_score': balanced_accuracy_score,
            'accuracy_score': accuracy_score,
            'roc_auc_score': roc_auc_score,
            'f2_score': partial(fbeta_score, beta=2),
            'f0.5_score': partial(fbeta_score, beta=0.5),
            'confusion_matrix': confusion_matrix,
            'tpr': tpr_score,
            'fpr': fpr_score,
            'AUC_zhu': compute_AUC_Zhu,
            'Warning_curve': warning_curve
        }

    def eval(
            self,
            metrics: list,
            unrolled_proba: dict,
            params_dict: dict,
            verbose: bool = True
    ):
        """
        Compute metrics on the shots given the threshold statistics.

        :param metrics: list of metrics to compute
        :param unrolled_proba: dictionary of unrolled probabilities
        :param params_dict: dictionary of parameters
        :param verbose: if True print metrics
        :return: metrics_report

        """

        # Convert from unrolled probabilities to [0,1] sequences
        self.unrolled_proba_to_shot_classif(unrolled_proba, params_dict)

        # Calculate metrics
        self.calc_metrics_report(
            metrics=metrics,
            unrolled_proba=unrolled_proba,
            params_dict=params_dict
        )

        # Print metrics if verbose
        if verbose:
            for k, v in self.metrics_report.items():
                logger.info(f'{k} - {np.round(v, 5)}')

        return self.metrics_report

    def calc_metrics_report(
            self,
            metrics: list,
            unrolled_proba: dict = None,
            params_dict: dict = None
    ):
        """
        Loops over metrics dict items and computes a metrics_report.

        :param metrics: list of metrics to compute
        :param unrolled_proba: unrolled probabilities
        :param params_dict: dictionary of parameters
        :return: metrics_report

        """

        self.metrics_report = {}
        for metric_name in metrics:
            if metric_name == 'Warning_curve':
                # Early warning curve
                try:
                    self.metrics_report.update(
                        {
                            metric_name: self.metrics_dict[metric_name](
                                class_shots_pred=self.class_shots_pred,
                                class_shots_true=self.class_shots_true,
                                time_alarm=self.time_alarm,
                                time_until_disrupt_alarm=self.time_until_disrupt_alarm

                            )
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f'Cannot eval metrics: pred {self.class_shots_pred.unique()}, true {self.class_shots_true.unique()} - {e}')
            elif metric_name == 'AUC_zhu':
                # Eval AUC ZHU
                try:
                    auc_zhu, best_thre, fpr_s, tpr_s = self.metrics_dict[metric_name](
                        unrolled_proba,
                        t_useful=params_dict['t_useful']
                    )

                    # Add tpr at 5 % of fpr
                    fpr_s_sort = np.array(fpr_s)[np.argsort(fpr_s)]
                    tpr_s_sort = np.array(tpr_s)[np.argsort(fpr_s)]
                    tpr_at_five = np.interp(0.05, fpr_s_sort, tpr_s_sort)
                    self.metrics_report.update(
                        {
                            'AUC_zhu': auc_zhu,
                            'best_thr_zhu': best_thre,
                            'roc_curve_zhu': [fpr_s_sort.tolist(), tpr_s_sort.tolist()],
                            'tpr_at_5%fpr_zhu': tpr_at_five
                        }
                    )
                except Exception as e:
                    logger.warning(f'Cannot eval {metric_name} - {e}')
            else:
                # Eval other metrics
                try:
                    self.metrics_report.update(
                        {
                            metric_name: self.metrics_dict[metric_name](self.class_shots_true, self.class_shots_pred)
                        }
                    )
                except Exception as e:
                    logger.info(f'Cannot eval {metric_name} - {e}')

    def unrolled_proba_to_shot_classif(
            self,
            unrolled_proba: dict,
            params_dict: dict,
    ):
        """
        From unrolled probabilities to shot classification using two-threshold+hysteresis time rule.

        :param unrolled_proba: dictionary of unrolled probabilities
        :param params_dict: dictionary of parameters

        """

        self.class_shots_pred = []
        self.class_shots_true = []
        self.time_alarm_lst = []
        self.time_until_disrupt_alarm_lst = []

        for idx_, shot in unrolled_proba.items():
            try:
                class_shots_pred, class_shots_true, time_alarm, time_until_disrupt_alarm = self.classify_shot(
                    **shot,
                    **params_dict
                )
                self.class_shots_pred.append(class_shots_pred)
                self.class_shots_true.append(class_shots_true)
                self.time_alarm_lst.append(time_alarm)
                self.time_until_disrupt_alarm_lst.append(time_until_disrupt_alarm)
            except TooShortShot:
                logger.warning(f'Shot {idx_} is too short')
                pass

        self.class_shots_pred = np.array(self.class_shots_pred)
        self.class_shots_true = np.array(self.class_shots_true)
        self.time_alarm = np.array(self.time_alarm_lst, dtype=float)
        self.time_until_disrupt_alarm = np.array(self.time_until_disrupt_alarm_lst, dtype=float)

    def classify_shot(
            self,
            proba_shot: np.ndarray,
            time_until_disrupt: np.ndarray,
            time_shot: np.ndarray,
            label_shot: int,
            high_thr: float,
            low_thr: float,
            t_hysteresis: float,
            t_useful: float,
            **kwargs
    ):
        """
        Single shot version of unrolled_proba_to_shot_classify. From unrolled probabilities
        to shot classification using two-threshold+hysteresis time rule.

        :param proba_shot: probability of being disruptive
        :param time_until_disrupt: time until disruption
        :param time_shot: time of the shot
        :param label_shot: label of the shot
        :param high_thr: high threshold
        :param low_thr: low threshold
        :param t_hysteresis: hysteresis time
        :param t_useful: useful time
        :param kwargs: other parameters
        :return: class_shots_pred, class_shots_true, time_alarm, time_until_disrupt_alarm

        """

        class_shots_true = label_shot
        time_alarm = np.nan
        time_until_disrupt_alarm = np.nan
        # if class_shots_true == 1:
        # cut last t_useful part of the shot, otherwise late a alarm would be a true positive
        # !!! not affecting AUC_zhu because it's using proba_shot directly !!!
        filter_ = time_until_disrupt >= t_useful
        proba_shot = proba_shot[filter_]
        time_shot = time_shot[filter_]
        time_until_disrupt = time_until_disrupt[filter_]

        # convert t_hysteresis [s] into hysteresis [number of points] by dividing time_step/t_hysteresis
        if t_hysteresis == 0:
            hysteresis = 0
        else:
            if time_shot.ravel().shape[0] > 1:
                hysteresis = (time_shot.ravel()[1] - time_shot.ravel()[0]).round(4) // t_hysteresis
            elif time_shot.ravel().shape[0] == 1:
                hysteresis = 0
            else:
                raise TooShortShot('Shot is too short')

        if high_thr == low_thr:
            # Disable hysteresis and look at the entire timeseries at once
            class_shots_pred = 1 if (proba_shot >= high_thr).any() else 0
            if class_shots_pred == 1:
                try:
                    time_alarm = time_shot[proba_shot >= high_thr][0]  # take the time when the alarm is raised
                except:
                    print()
                # Take the time until disruption when the alarm is raised
                time_until_disrupt_alarm = time_until_disrupt[proba_shot >= high_thr][
                    0]

        else:

            # For speeding-up the computation: if none of proba_shot is above high_thr,
            # label the shot as non-disruptive, else loop over all the timeseries points
            if (proba_shot < high_thr).all():
                class_shots_pred = 0
            else:
                # Initialize hysteresis counter
                hysteresis_counter = 0
                class_shots_pred = 0
                for i in range(proba_shot.shape[0]):
                    if proba_shot[i] >= high_thr:
                        hysteresis_counter += 1
                    elif proba_shot[i] < low_thr:
                        hysteresis_counter = 0

                    if hysteresis_counter > hysteresis:
                        class_shots_pred = 1
                        time_alarm = time_shot[i]
                        # Take the time until disruption when the alarm is raised
                        time_until_disrupt_alarm = time_until_disrupt[i]
                        # If the alarm is raised, exit the loop
                        break

        return class_shots_pred, class_shots_true, time_alarm, time_until_disrupt_alarm
