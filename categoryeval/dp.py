from typing import List, Union, Tuple, Dict, Optional
import numpy as np
from pyitlib import discrete_random_variable as drv

from categoryeval.probestore import ProbeStore
from categoryeval.representation import make_context_by_term_matrix


class DPScorer:
    """
    computes divergence-from-prototype, a measure of how close a learned representation is to a category prototype.
    """

    def __init__(self,
                 probe2cat: Dict[str, str],
                 tokens: List[str],
                 ) -> None:

        print('Initializing DPScorer...')
        self.probe_store = ProbeStore(probe2cat)

        # make p for each name - p is a theoretical probability distribution over x-words (next-words)
        # rows index contexts, and columns index next-words
        self.ct_mat, self.x_words, y_words_ = make_context_by_term_matrix(tokens, context_size=1)
        self.total_frequency = self.ct_mat.sum().sum().item()
        self.y_words = [self.tuple2str(yw) for yw in y_words_]  # convert tuple to str
        self.p = self._make_p()

    def calc_dp(self,
                qs: np.ndarray,
                return_mean: bool = True,
                metric: str = 'js'
                ) -> Union[float, List[float]]:
        """
        measure bits divergence of a set of next-word predictions from prototype next-word probability distribution,
        where the prototype is the category to which all probes labeled "probes_name" belong.
        dp = divergence-from-prototype

        "p" is a true probability distribution
        "q" is an approximation
        """
        assert np.ndim(qs) == 2
        assert np.sum(qs[0]).round(1).item() == 1.0, np.sum(qs[0]).round(1).item()

        if metric == 'ce':
            raise NotImplementedError
        elif metric == 'xe':
            fn = drv.entropy_cross_pmf
        elif metric == 're':
            raise NotImplementedError
        elif metric == 'js':
            fn = drv.divergence_jensenshannon_pmf
        else:
            raise AttributeError('Invalid arg to "metric".')

        # compare each word's predicted and expected next-word probability distribution
        p = self.p
        res = [fn(p, q) for q in qs]

        if return_mean:
            return np.mean(res).item()
        else:
            return res

    @staticmethod
    def tuple2str(yw):
        """convert a context (one or more y-words) which is an instance of a tuple to a string"""
        return '_'.join(yw)

    def _make_p(self,
                is_unconditional: bool = False,
                e=0.00000000001,  # probabilities cannot be zero -otherwise cross-entropy is inf
                ) -> np.ndarray:
        """
        make the true next-word probability distribution (by convention called "p"),
        which is defined as each word representing an iid sample from the distribution.
        """

        if is_unconditional:  # "unconditional" is a category whose members are all words in vocab
            return self._make_unconditional_p()

        # get slice of ct matrix
        probes = self.probe_store.types
        row_ids = [self.y_words.index(w) for w in probes]
        assert row_ids
        sliced_ct_mat = self.ct_mat.tocsc()[row_ids, :]

        # make p, the true probability distribution over y-words given some category.
        # assumes there is a single distribution generating all probes
        res = []
        for col_id, xw in enumerate(self.x_words):
            if xw in probes:  # a test-word is not allowed to be next-word of a test-word
                f = e
            else:
                xw_frequency = sliced_ct_mat[:, col_id].sum()
                if xw_frequency > 0.0:
                    f = xw_frequency
                else:
                    f = e
            res.append(f)
        return np.array(res) / np.sum(res)

    def _make_unconditional_p(self
                              ) -> np.ndarray:
        """
        make theoretical next-word distribution for the "average" word;
        that is, what is the best next-word distribution given any word from the vocabulary,
         not just members of a category?
        """
        res = self.ct_mat.tocsr().sum(axis=0).A1 / self.total_frequency  # A1 converts matrix to 1D array
        return res
