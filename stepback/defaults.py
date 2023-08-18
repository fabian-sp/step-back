class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


_defaults = {'config_dir': 'configs/',
            'output_dir': 'output/',
            'data_dir': 'data/',
            'device': 'cuda',
            'num_workers': 0,
            'data_parallel': None,
            'verbose': False,
            'force_deterministic': False
            }

DEFAULTS = Dotdict(_defaults)