from cached_property import cached_property
from sortedcontainers import SortedSet

from categoryeval import config


class ProbeStore(object):
    """
    Stores probe-related data.
    """

    def __init__(self, corpus_name, probes_name, w2id=None):
        self.corpus_name = corpus_name
        self.probes_name = probes_name
        self.w2id = w2id

        self.file_name = f'{corpus_name}_{len(w2id)}_{probes_name}.txt'
        print(f'Initialized probe_store from {self.file_name}')

    @cached_property
    def probe2cat(self):
        probe2cat = {}
        p = config.Dirs.probes / self.file_name
        with p.open('r') as f:
            for line in f:
                data = line.strip().strip('\n').split()
                probe = data[0]
                cat = data[1]
                if self.w2id is not None:
                    if probe not in self.w2id:
                        print(f'WARNING: Probe "{probe}" not in vocabulary -> Excluded from analysis')
                    else:
                        probe2cat[probe] = cat
                else:
                    probe2cat[probe] = cat
        return probe2cat

    @cached_property
    def types(self):
        probes = sorted(self.probe2cat.keys())
        probe_set = SortedSet(probes)
        print('Num probes: {}'.format(len(probe_set)))
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
        cat2probes = {cat: [probe for probe in self.types if self.probe2cat[probe] == cat]
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
