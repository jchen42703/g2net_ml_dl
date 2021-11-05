import numpy as np
import torch
'''
Dataset
'''


class G2NetDataset(torch.utils.data.Dataset):
    '''
    Amplitude stats
    [RAW]
    max of max: [4.6152116e-20, 4.2303907e-20, 1.1161064e-20]
    mean of max: [1.8438003e-20, 1.8434544e-20, 5.0978556e-21]
    max of mean: [1.5429503e-20, 1.5225015e-20, 3.1584522e-21]
    [BANDPASS]
    max of max: [1.7882743e-20, 1.8305723e-20, 9.5750025e-21]
    mean of max: [7.2184587e-21, 7.2223450e-21, 2.4932809e-21]
    max of mean: [6.6964011e-21, 6.4522511e-21, 1.4383649e-21]
    '''

    def __init__(
        self,
        paths,
        targets=None,
        transforms=None,
        mixup=False,
        mixup_alpha=0.4,
        mixup_option='random',
        hard_label=False,
        lor_label=False,
        is_test=False,
    ):
        self.paths = paths
        self.targets = targets
        self.negative_idx = np.where(self.targets == 0)[0]
        self.transforms = transforms
        self.mixup = mixup
        self.alpha = mixup_alpha
        self.mixup_option = mixup_option
        self.hard_label = hard_label
        self.lor_label = lor_label
        self.is_test = is_test
        if self.is_test:
            self.mixup = False

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """Called in every iteration to generate a batch of data.
        """
        signal, target = self._get_signal_target(index)
        if self.mixup:
            signal, target = self.apply_mixup(signal, target)

        return (signal, target)

    def _get_signal_target(self, index: int):
        """Loads the signal as a numpy array, applies the transforms and returns
        the labels if they exist.

        Args:
            index: the index of the paths array that you want to load a np array
                from
        """
        path = self.paths[index]
        signal = np.load(path).astype(np.float32)

        if self.transforms is not None:
            signal = self.transforms(signal.copy())

        if self.targets is not None:
            target = torch.tensor(self.targets[index]).unsqueeze(0).float()
        else:
            target = torch.tensor(0).unsqueeze(0).float()
        return (signal, target)

    def apply_mixup(self, signal, target):
        """Applies the mixup transformation.
        """
        if self.mixup_option == 'random':
            idx2 = np.random.randint(0, len(self))
            lam = np.random.beta(self.alpha, self.alpha)
        elif self.mixup_option == 'negative':
            idx2 = np.random.randint(0, len(self.negative_idx))
            idx2 = self.negative_idx[idx2]
            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1 - lam)  # negative noise is always weaker
        signal2, target2 = self._get_signal_target(idx2)
        signal = lam * signal + (1 - lam) * signal2
        if self.lor_label:
            target = target + target2 - target * target2
        else:
            target = lam * target + (1 - lam) * target2
            if self.hard_label:
                target = (target > self.hard_label).float()
        return (signal, target)