from typing import List, Dict, Set, Optional
import numpy as np
from pyitlib import discrete_random_variable as drv

from categoryeval import configs
from categoryeval.probestore import ProbeStore


class RAScorer:
    def __init__(self,
                 probe2cat: Dict[str, str],
                 ) -> None:

        print('Initializing RAScorer...')
        self.probe_store = ProbeStore(probe2cat)

    def calc_score(self,
                   ps: np.ndarray,
                   qs: np.ndarray,
                   metric: str = 'js',
                   max_rows: int = 32,
                   ):
        """
        Compute divergence between two probability distributions (output by some model)
        given two inputs that are nearby in the input space.

        intuition: if two inputs (which are nearby) produce large divergences at the output,
        it can be said that the model's input-output mapping is ragged.
        """
        print(f'Computing raggedness...')

        assert len(ps) == len(qs)

        rand_ids = np.random.randint(0, len(ps), max_rows)
        ps_sample = ps[rand_ids]
        qs_sample = qs[rand_ids]

        if metric == 'xe':
            return drv.entropy_cross_pmf(ps_sample, qs_sample, base=2, cartesian_product=False).mean()
        elif metric == 'js':
            return drv.divergence_jensenshannon_pmf(ps_sample, qs_sample, base=2, cartesian_product=False).mean()
        else:
            raise AttributeError('Invalid arg to "metric".')