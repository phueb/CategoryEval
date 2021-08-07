from typing import List, Dict, Set, Optional
import numpy as np

from categoryeval.s_dbw import S_Dbw
from categoryeval.probestore import ProbeStore


class SDScorer:
    def __init__(self,
                 probe2cat: Dict[str, str],
                 ) -> None:

        self.probe_store = ProbeStore(probe2cat)

    def calc_sd(self,
                representations: np.array,
                category_labels: List[int],
                metric: str = 'cosine',
                ):
        """
        using code from https://github.com/alashkov83/S_Dbw
        """

        return S_Dbw(representations,
                     category_labels,
                     centers_id=None,
                     method='Tong',
                     alg_noise='bind',
                     centr='mean',
                     nearest_centr=True,
                     metric=metric)
