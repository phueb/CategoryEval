from typing import List, Dict, Set, Optional
import numpy as np
from pyitlib import discrete_random_variable as drv

from categoryeval.probestore import ProbeStore


class CSScorer:
    """
    computes category-separation:
    a measure of how close next-word probability distributions for one category are to those of another
    """

    def __init__(self,
                 probe2cat: Dict[str, str],
                 ) -> None:

        print('Initializing CSScorer...')

        self.probe_store = ProbeStore(probe2cat)

    def calc_cs(self,
                ps: np.ndarray,
                qs: np.ndarray,
                metric: str = 'js',
                max_rows: int = 32,
                ) -> float:
        """
        measure bits divergence of a set of next-word predictions from another set of next-word predictions

        "p" is a true probability distribution
        "q" is an approximation
        """
        assert np.ndim(qs) == 2
        assert np.sum(qs[0]).round(1).item() == 1.0, np.sum(qs[0]).round(1).item()

        print('Computing cs...')

        ps_sample = ps[np.random.choice(len(ps), size=min(len(ps), len(qs), max_rows), replace=False)]
        qs_sample = qs[np.random.choice(len(qs), size=min(len(ps), len(qs), max_rows), replace=False)]

        if metric == 'xe':
            return drv.entropy_cross_pmf(ps_sample, qs_sample, base=2, cartesian_product=True).mean()
        elif metric == 'js':
            return drv.divergence_jensenshannon_pmf(ps_sample, qs_sample, base=2, cartesian_product=True).mean()
        else:
            raise AttributeError('Invalid arg to "metric".')