import numpy as np


def adjust_context(windows, context_type):
    if context_type == 'none':
        x = windows[:, -1][:, np.newaxis]
    elif context_type == 'ordered':
        x = windows
    elif context_type == 'only':
        x = windows[:, :-1]
    elif context_type == 'last':
        x = windows[:, -2][:, np.newaxis]
    elif context_type == 'shuffled':
        x_no_probe = windows[:, np.random.permutation(np.arange(windows.shape[1] - 1))]
        x = np.hstack((x_no_probe, np.expand_dims(windows[:, -1], axis=1)))
    else:
        raise AttributeError('Invalid arg to "context_type".')
    return x
