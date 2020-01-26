from cytoolz import itertoolz
from typing import List


def split(l, split_size):
    for i in range(0, len(l), split_size):
        yield l[i:i + split_size]


def get_sliding_windows(window_size: int,
                        tokens: List[str],
                        ) -> List[List[str]]:
    res = list(itertoolz.sliding_window(window_size, tokens))
    return res