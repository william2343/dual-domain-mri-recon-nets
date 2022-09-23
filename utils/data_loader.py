import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SingleFileDataset(Dataset):
    def __init__(self, gt_path, input_path=None, undersample_fn=None, transform=None):
        """
        A dataset from a single numpy file.
        """
        self.gt = self._load_from_path(gt_path)
        self.input = self._load_from_path(input_path) if input_path else None # currently does nothing
        self.undersample_fn = undersample_fn
        self.transform = transform

    def __len__(self):
        return len(self.gt) - 1  # To handle list1.npy  # TODO: round down to next multiple of 50.

    def __getitem__(self, index):
        start, end = self.select_5_frames(index, sequence_len=50)
        y = self.gt[start:end]
        # y = np.clip(y, a_min=0, a_max=255) / 255
        y = (y/y.max())

        if self.transform:
            y = self.transform(y) 

        if self.undersample_fn:
            x = self.undersample_fn(y)
            return x, y
        else:
            return y

    @staticmethod
    def _load_from_path(path):
        return torch.from_numpy(np.load(path).astype(np.float32))

    @staticmethod
    def select_5_frames(index, sequence_len=50):
        idx = np.clip(index % sequence_len, 2, (sequence_len-3)).astype(int)
        start = (idx - 2)
        end = (idx + 3)
        seq_idx = int(index//sequence_len) * sequence_len
        return seq_idx + start, seq_idx + end

if __name__ == "__main__":
    # torch.multiprocessing.freeze_support()
    path = "dataset_uint.npy"
    def undersampler(tensor):
        return tensor

    dataset = SingleFileDataset(
    gt_path=path,
    input_path=None,
    undersample_fn=undersampler,
    transform=None
    )

    loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2
    )

    for x, y in loader:
        print(f"x: {x.shape}")
        print(f"y: {y.shape}")
        break