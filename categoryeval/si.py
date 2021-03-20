from typing import List, Dict, Set, Optional
import numpy as np
from sklearn.metrics.cluster import silhouette_score, calinski_harabasz_score

from categoryeval.probestore import ProbeStore


class SIScorer:
    def __init__(self,
                 corpus_name: str,
                 probes_names: List[str],
                 excluded: Optional[Set[str]] = None,
                 ) -> None:

        print('Initializing SIScorer...')

        assert len(probes_names) == len(set(probes_names))

        self.probes_names = probes_names
        self.name2store = {probes_name: ProbeStore(corpus_name, probes_name, excluded)
                           for probes_name in probes_names}

    def calc_si(self,
                representations: np.array,
                category_labels: List[int],
                metric: str = 'cosine',
                ):
        """
        """
        print(f'Computing silhouette scores...')

        return silhouette_score(representations, category_labels, metric)