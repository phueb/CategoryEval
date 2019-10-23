from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    probes = root / 'probes'


class Eval:
    num_opt_init_steps = 5
    num_opt_steps = 10
    xi = 0.01  # 0.01 is better than 0.05
    verbose = True
    eval_thresholds = [[0.99], [0.9]]