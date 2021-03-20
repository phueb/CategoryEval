from cached_property import cached_property
from sortedcontainers import SortedSet
import numpy as np
from typing import List, Optional, Dict, Set

from categoryeval import configs


class ProbeStore(object):
    """
    Stores probe-related data.
    """

    def __init__(self,
                 corpus_name: str,
                 probes_name: str,
                 excluded: Optional[Set] = None,
                 warn: bool = True,
                 ):
        self.corpus_name = corpus_name
        self.probes_name = probes_name

        self.excluded = {} if excluded is None else excluded
        self.warn = warn

    @cached_property
    def probe2cat(self):
        probe2cat = {}
        p = configs.Dirs.probes / self.corpus_name / f'{self.probes_name}.txt'
        num_total = 0
        with p.open('r') as f:
            for line in f:
                data = line.strip().strip('\n').split()
                probe = data[0]
                cat = data[1]

                if probe in self.excluded:
                    if self.warn:
                        print(f'WARNING: Probe {probe: <12} in excluded list  -> Excluded from analysis')
                else:
                    probe2cat[probe] = cat
                num_total += 1

        print('Num probes loaded={}'.format(len(probe2cat)))

        return probe2cat

    @cached_property
    def types(self):
        probes = sorted(self.probe2cat.keys())
        probe_set = SortedSet(probes)
        return probe_set

    @cached_property
    def probe2id(self):
        probe2id = {probe: n for n, probe in enumerate(self.types)}
        return probe2id

    @cached_property
    def cats(self):
        cats = sorted(self.probe2cat.values())
        cat_set = SortedSet(cats)
        return cat_set

    @cached_property
    def cat2id(self):
        cat2id = {cat: n for n, cat in enumerate(self.cats)}
        return cat2id

    @cached_property
    def cat2probes(self):
        cat2probes = {cat: {probe for probe in self.types if self.probe2cat[probe] == cat}
                      for cat in self.cats}
        return cat2probes

    @cached_property
    def num_probes(self):
        num_probes = len(self.types)
        return num_probes

    @cached_property
    def num_cats(self):
        num_cats = len(self.cats)
        return num_cats

    # //////////////////////////////////////////////// for evaluation

    @cached_property
    def gold_sims(self):
        """
        returns binary matrix of shape [num_probes, num_probes] which defines the category structure.
        used for evaluation.
        """
        num_rows = self.num_probes
        num_cols = self.num_probes
        res = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            probe1 = self.types[i]
            for j in range(num_cols):
                probe2 = self.types[j]
                if self.probe2cat[probe1] == self.probe2cat[probe2]:
                    res[i, j] = 1
        return res.astype(np.bool)
