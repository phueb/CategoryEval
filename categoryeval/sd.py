from typing import List, Dict, Set, Optional
import numpy as np

from categoryeval.s_dbw import S_Dbw
from categoryeval.probestore import ProbeStore


class SDScorer:
    def __init__(self,
                 corpus_name: str,
                 probes_names: List[str],
                 w2id: Dict[str, int],
                 excluded: Optional[Set[str]] = None,
                 ) -> None:

        print('Initializing SDScorer...')

        assert len(probes_names) == len(set(probes_names))

        self.probes_names = probes_names
        self.name2store = {probes_name: ProbeStore(corpus_name, probes_name, w2id, excluded)
                           for probes_name in probes_names}

    def calc_sd(self,
                representations: np.array,
                category_labels: List[int],
                metric: str = 'cosine',
                ):
        """
        using code from https://github.com/alashkov83/S_Dbw
        """
        print(f'Computing S-Dbw score...')

        return S_Dbw(representations,
                     category_labels,
                     centers_id=None,
                     method='Tong',
                     alg_noise='bind',
                     centr='mean',
                     nearest_centr=True,
                     metric=metric)
