from typing import List, Dict, Set, Optional
import numpy as np
from bayes_opt import BayesianOptimization
from functools import partial

from categoryeval import configs


class BAScorer:
    def __init__(self,
                 probe2cat: Dict[str, str],
                 ) -> None:

        self.probe2cat = probe2cat

        self.gold_sims = self.make_gold_sims()

    def calc_score(self,
                   pred_sims: np.array,
                   gold_sims: np.array,
                   metric: str = 'ba',
                   return_threshold: bool = False,
                   ) -> float:
        """
        pred_sims is matrix of floats with shape [num_probes, num_probes]
        gold_sims is matrix of integers with shape [num_probes, num_probes]
        """
    
        if pred_sims.shape != gold_sims.shape:
            raise ValueError(f'Shape of predicted={pred_sims.shape} does not match gold similarities ={gold_sims.shape}')
    
        def calc_signals(_probe_sims, _labels, thr):  # vectorized algorithm is 20X faster
            probe_sims_clipped = np.clip(_probe_sims, 0, 1)
            probe_sims_clipped_triu = probe_sims_clipped[np.triu_indices(len(probe_sims_clipped), k=1)]
            predictions = np.zeros_like(probe_sims_clipped_triu, int)
            predictions[np.where(probe_sims_clipped_triu > thr)] = 1
            #
            tp = float(len(np.where((predictions == _labels) & (_labels == 1))[0]))
            tn = float(len(np.where((predictions == _labels) & (_labels == 0))[0]))
            fp = float(len(np.where((predictions != _labels) & (_labels == 0))[0]))
            fn = float(len(np.where((predictions != _labels) & (_labels == 1))[0]))
            return tp, tn, fp, fn
    
        # define calc_signals_partial
        self.check_nans(pred_sims, name='probe_sims')
        labels = gold_sims[np.triu_indices(len(gold_sims), k=1)]
        calc_signals_partial = partial(calc_signals, pred_sims, labels)
    
        def calc_probes_fs(thr):
            """
            WARNING: this gives incorrect results at early timepoints (lower compared to tensorflow implementation)
            # TODO this not due to using sim_mean as first point to bayesian-opt:
            # TODO perhaps exploration-exploitation settings are only good for ba but not f1
    
            """
            tp, tn, fp, fn = calc_signals_partial(thr)
            precision = np.divide(tp + 1e-7, (tp + fp + 1e-7))
            sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
            fs = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-7)
            return fs
    
        def calc_probes_ck(thr):
            """
            cohen's kappa
            """
            tp, tn, fp, fn = calc_signals_partial(thr)
            totA = np.divide(tp + tn, (tp + tn + fp + fn))
            #
            pyes = ((tp + fp) / (tp + fp + tn + fn)) * ((tp + fn) / (tp + fp + tn + fn))
            pno = ((fn + tn) / (tp + fp + tn + fn)) * ((fp + tn) / (tp + fp + tn + fn))
            #
            randA = pyes + pno
            ck = (totA - randA) / (1 - randA)
            # print('totA={:.2f} randA={:.2f}'.format(totA, randA))
            return ck
    
        def calc_probes_ba(thr):
            tp, tn, fp, fn = calc_signals_partial(thr)
            specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
            sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
            ba = (sensitivity + specificity) / 2  # balanced accuracy
            return ba
    
        # use bayes optimization to find best_thr
        if metric == 'f1':
            fun = calc_probes_fs
        elif metric == 'ba':
            fun = calc_probes_ba
        elif metric == 'ck':
            fun = calc_probes_ck
        else:
            raise AttributeError('Invalid arg to "cluster_metric".')
        bo = BayesianOptimization(fun, {'thr': (0.0, 1.0)}, verbose=configs.BA.verbose)
        bo.init_points.extend(configs.BA.eval_thresholds + [[pred_sims.mean()]])
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
        bo.maximize(init_points=configs.BA.num_opt_init_steps, n_iter=configs.BA.num_opt_steps,
                    acq="poi", xi=configs.BA.xi, **gp_params)  # smaller xi: exploitation
        best_thr = bo.res['max']['max_params']['thr']
        # use best_thr
        results = fun(best_thr)
        res = np.mean(results)

        if return_threshold:
            return res, best_thr
        else:
            return res

    def make_gold_sims(self):
        """
        returns binary matrix of shape [num_probes, num_probes] which defines the category structure.
        used for evaluation.
        """

        probes = sorted(self.probe2cat.keys())
        num_rows = len(probes)
        num_cols = len(probes)
        res = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            probe1 = probes[i]
            for j in range(num_cols):
                probe2 = probes[j]
                if self.probe2cat[probe1] == self.probe2cat[probe2]:
                    res[i, j] = 1
        return res.astype(np.bool)

    def check_nans(self, mat, name='unnamed'):
        if np.any(np.isnan(mat)):
            num_nans = np.sum(np.isnan(mat))
            print(f'Found {num_nans} Nans in {name}')