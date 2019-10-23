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


def make_probe_prototype_acts_mat(context_type):
    print('Making probe prototype activations...')
    res = np.zeros((hub.probe_store.num_probes, hub.params.embed_size))
    for n, probe_x_mat in enumerate(hub.probe_x_mats):
        x = adjust_context(probe_x_mat, context_type)
        # probe_act
        probe_exemplar_acts_mat = sess.run(h, feed_dict={graph.x: x})
        probe_prototype_act = np.mean(probe_exemplar_acts_mat, axis=0)
        res[n] = probe_prototype_act
    return res