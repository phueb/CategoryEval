from typing import List, Optional

from scipy import sparse
from sortedcontainers import SortedSet

from categoryeval.utils import get_sliding_windows


def make_context_by_term_matrix(tokens: List[str],
                                x_words: List[str],
                                context_size: Optional[int] = 1
                                ):
    """
    terms are in cols, contexts are in rows.
    y_words are contexts (possibly multi-word), x_words are targets (always single-word).
    a context precedes a target.
    """

    print('Making context-term matrix...')

    num_xws = len(x_words)
    xw2col_id = {t: n for n, t in enumerate(x_words)}

    # contexts
    contexts_in_order = get_sliding_windows(context_size, tokens)
    y_words = SortedSet(contexts_in_order)
    num_y_words = len(y_words)
    yw2row_id = {c: n for n, c in enumerate(y_words)}

    # make sparse matrix (contexts/y-words in rows, targets/x-words in cols)
    data = []
    row_ids = []
    cold_ids = []
    for n, context in enumerate(contexts_in_order[:-context_size]):
        # row_id + col_id
        row_id = yw2row_id[context]
        next_context = contexts_in_order[n + 1]
        next_word = next_context[-1]  # -1 is correct because windows slide by 1 word
        try:
            col_id = xw2col_id[next_word]
        except KeyError:  # when probe_store is passed, only probes are n xw2col_id
            continue
        # collect
        row_ids.append(row_id)
        cold_ids.append(col_id)
        data.append(1)  # it is okay to append 1s because final value is sum over 1s in same position in matrix

    # make sparse matrix once (updating it is expensive)
    res = sparse.coo_matrix((data, (row_ids, cold_ids)))

    print(f'Co-occurrence matrix has sum={res.sum():,} and shape={res.shape}')
    expected_shape = (num_y_words, num_xws)
    if res.shape != expected_shape:
        raise SystemExit(f'Result does not match expected shape={expected_shape}')

    return res, y_words