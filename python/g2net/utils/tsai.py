import torch
import time
import warnings
from datetime import datetime
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys

IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    from numba import config
    config.THREADING_LAYER = 'omp'


def to_local_time(t, time_format='%Y-%m-%d %H:%M:%S'):
    return time.strftime(time_format, time.localtime(t))


def print_verbose(message: str, verbose: bool = True):
    if verbose:
        print(message)


class Timer:

    def start(self, verbose=True):
        self.all_elapsed = 0
        self.n = 0
        self.verbose = verbose
        self.start_dt = datetime.now()
        self.start_dt0 = self.start_dt

    def elapsed(self):
        end_dt = datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0:
            self.all_elapsed = elapsed
        else:
            self.all_elapsed += elapsed
        print_verbose(f'Elapsed time ({self.n:3}): {elapsed}', self.verbose)
        self.start_dt = datetime.now()
        if not self.verbose:
            return elapsed

    def stop(self):
        end_dt = datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0:
            self.all_elapsed = elapsed
        else:
            self.all_elapsed += elapsed
        total_elapsed = end_dt - self.start_dt0
        delattr(self, "start_dt0")
        delattr(self, "start_dt")
        if self.verbose:
            if self.n > 1:
                print(f'Elapsed time ({self.n:3}): {elapsed}')
                print(f'Total time        : {self.all_elapsed}')
            else:
                print(f'Total time        : {total_elapsed}')
        else:
            return total_elapsed


def computer_setup():
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import platform
        print(f'os             : {platform.platform()}')
    except:
        pass
    try:
        from platform import python_version
        print(f'python         : {python_version()}')
    except:
        pass
    try:
        import tsai
        print(f'tsai           : {tsai.__version__}')
    except:
        print(f'tsai           : N/A')
    try:
        import fastai
        print(f'fastai         : {fastai.__version__}')
    except:
        print(f'fastai         : N/A')
    try:
        import fastcore
        print(f'fastcore       : {fastcore.__version__}')
    except:
        print(f'fastcore       : N/A')

    try:
        import torch
        print(f'torch          : {torch.__version__}')
        try:
            import torch_xla
            print(f'device         : TPU')
        except:
            cpus = os.cpu_count()
            iscuda = torch.cuda.is_available()
            print(f'n_cpus         : {cpus}')
            print(f'device         : {device} ({torch.cuda.get_device_name(0)})'
                  if iscuda else f'device         : {device}')
    except:
        print(f'torch          : N/A')
