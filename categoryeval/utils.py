from cytoolz import itertoolz
from typing import List, Dict, Set



def get_sliding_windows(window_size: int,
                        tokens: List[str],
                        ) -> List[List[str]]:
    res = list(itertoolz.sliding_window(window_size, tokens))
    return res