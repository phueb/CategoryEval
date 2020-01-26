from typing import List, Union
import numpy as np
from collections import Counter

from categoryeval.representation import make_context_by_term_matrix
from categoryeval.utils import split
from categoryeval import config


class DPScorer:
    """
    computes divergence-from-prototype, a measure of how close a learned representation is to a category prototype.
    """

    def __init__(self,
                 corpus_name: str,
                 probes_names: List[str],  # a list of names for files with probe words
                 tokens: List[str],  # the tokens which will be used for computing prototype representation
                 types: List[str],
                 num_parts: int,  # number of parts to split tokens - required to create name2probe2part
                 ) -> None:

        print('Initializing DPScorer...')

        assert len(probes_names) == len(set(probes_names))

        self.corpus_name = corpus_name
        self.probes_names = probes_names
        self.num_parts = num_parts
        self.name2probes = {name: types if name == 'unconditional' else (self.load_probes(corpus_name, name))
                            for name in self.probes_names}

        # make q for each name - q is a theoretical probability distribution over x-words
        # rows index contexts, and columns index next-words
        self.ct_mat, self.x_words, y_words_ = make_context_by_term_matrix(tokens, context_size=1)
        self.total_frequency = self.ct_mat.sum().sum().item()
        self.y_words = [self.tuple2str(yw) for yw in y_words_]  # convert tuple to str
        self.name2q = {name: self._make_q(name) for name in self.probes_names}

        # make name2part2probes - assign probes to a corpus partition
        split_size = len(tokens) // num_parts
        part2w2f = {n: Counter(tokens_part) for n, tokens_part in enumerate(split(tokens, split_size))}
        self.name2part2probes = {name: {part: [] for part in range(num_parts)}
                                 for name in self.probes_names}
        for name, probes in self.name2probes.items():
            for probe in probes:
                part = np.argmax([part2w2f[part][probe] for part in range(num_parts)])
                self.name2part2probes[name][part].append(probe)

    def calc_dp(self,
                predictions_mat: np.ndarray,
                probes_name: str,
                return_mean: bool = True,
                return_sim: bool = False,
                ) -> Union[float, List[float]]:
        """
        measure bits divergence of a set of next-word predictions from prototype next-word probability distribution,
        where the prototype is the category to which all probes labeled "probes_name" belong.
        dp = distance-to-prototype
        """
        assert np.ndim(predictions_mat) == 2
        assert np.sum(predictions_mat[0]).round(1).item() == 1.0, np.sum(predictions_mat[0]).round(1).item()

        if return_sim:
            fn = self._pearson
        else:
            fn = self._calc_kl_divergence

        # kld divergence between each word's predicted and expected prob distribution over context
        q = self.name2q[probes_name]
        klds = [fn(p, q) for p in predictions_mat]

        if return_mean:
            return np.mean(klds).item()
        else:
            return klds

    @staticmethod
    def _pearson(p: np.ndarray,
                 q: np.ndarray,
                 ) -> float:

        return np.corrcoef(p, q)[0, 1].item()

    @staticmethod
    def _calc_kl_divergence(p: np.ndarray,
                            q: np.ndarray,
                            epsilon=0.00001,
                            ) -> float:
        assert len(p) == len(q), (len(p), len(q))

        pe = p + epsilon
        qe = q + epsilon
        res = np.sum(pe * np.log2(pe / qe)).item()
        return res

    @staticmethod
    def load_probes(corpus_name: str,
                    probes_name: str
                    ) -> List[str]:
        p = config.Dirs.probes / corpus_name / 'dp' / f'{probes_name}.txt'
        res = p.read_text().split('\n')
        assert len(res) == len(set(res))
        return res

    @staticmethod
    def tuple2str(yw):
        """convert a context (one or more y-words) which is an instance of a tuple to a string"""
        return '_'.join(yw)

    def _make_q(self,
                probes_name: str,
                ) -> np.ndarray:
        """
        make theoretical next-word probability distribution,
        which is defined as each word representing an iid sample from the distribution.
        """

        if probes_name == 'unconditional':  # "unconditional" is a category whose members are all words in vocab
            return self._make_unconditional_q()

        # get slice of ct matrix
        probes = self.name2probes[probes_name]
        row_ids = [self.y_words.index(w) for w in probes]
        assert row_ids
        sliced_ct_mat = self.ct_mat.tocsc()[row_ids, :]
        slice_ct_mat_sum = sliced_ct_mat.sum().sum().item()

        # make q, the probability distribution over y-words given some category.
        # assumes there is a single distribution generating all probes
        res = []
        for col_id, xw in enumerate(self.x_words):
            if xw in probes:  # a test-word is not allowed to be next-word of a test-word
                prob = 0.0
            else:
                xw_frequency = sliced_ct_mat[:, col_id].sum()
                if xw_frequency > 0.0:
                    prob = xw_frequency / slice_ct_mat_sum
                else:
                    prob = 0.0
            res.append(prob)
        return np.array(res)

    def _make_unconditional_q(self
                        ) -> np.ndarray:
        """
        make theoretical next-word distribution for the "average" word;
        that is, what is the best next-word distribution given any word from the vocabulary,
         not just members of a category?
        """
        res = self.ct_mat.tocsr().sum(axis=0).A1 / self.total_frequency  # A1 converts matrix to 1D array
        return res
