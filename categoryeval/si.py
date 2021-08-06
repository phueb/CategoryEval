from typing import List, Dict, Set, Optional
import numpy as np
from sklearn.metrics.cluster import silhouette_score, calinski_harabasz_score

from categoryeval.probestore import ProbeStore


class SIScorer:
    def __init__(self,
                 probe2cat: Dict[str, str],
                 ) -> None:

        print('Initializing SIScorer...')
        self.probe_store = ProbeStore(probe2cat)

    def calc_si(self,
                representations: np.array,
                category_labels: List[int],
                metric: str = 'cosine',
                ):
        """
        """

        return silhouette_score(representations, category_labels, metric)