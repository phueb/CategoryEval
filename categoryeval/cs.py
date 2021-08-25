from typing import List, Dict, Set, Optional
import numpy as np
from pyitlib import discrete_random_variable as drv


class CSScorer:
    """
    computes category-separation:
    a measure of how close next-word probability distributions for one category are to those of another
    """

    def __init__(self,
                 ) -> None:
        pass


    def calc_score(self,
                   ps: np.ndarray,
                   qs: np.ndarray,
                   metric: str = 'js',  # 'xe' is faster than 'js' but not normalized
                   max_rows: int = 32,
                   ) -> float:
        """
        measure bits divergence of a set of next-word predictions from another set of next-word predictions

        note: when using cross-entropy, argument ordering matters:
            "p" is a true probability distribution
            "q" is an approximation

        note:
            computation of divergence checks for NaNs. this is done in 2 ways, slow or fast, depending on dtype:
            if dtype is float64, fast check is performed (float32 results in slow check)
            the slow check is much slower and should be avoided.
        """
        assert np.ndim(ps) == 2
        assert np.ndim(qs) == 2
        assert ps.shape == qs.shape
        assert np.sum(qs[0]).round(1).item() == 1.0, np.sum(qs[0]).round(1).item()

        if ps.dtype != np.float64 or qs.dtype != np.float64:
            raise TypeError('To aovid slow NaN check, cast input to CSScorer.calc_score to float64.')

        if max_rows < len(qs):
            print(f'Randomly sampling input because max_rows={max_rows} < num rows in input={len(qs)}', flush=True)
            ps = ps[np.random.choice(len(ps), size=max_rows, replace=False)]
            qs = qs[np.random.choice(len(qs), size=max_rows, replace=False)]

        if metric == 'xe':
            return drv.entropy_cross_pmf(ps, qs, base=2, cartesian_product=True).mean()
        elif metric == 'js':
            return drv.divergence_jensenshannon_pmf(ps, qs, base=2, cartesian_product=True).mean()
        else:
            raise AttributeError('Invalid arg to "metric".')