import abc
from functools import partial
from glob import glob
import random
from time import perf_counter
from typing import Tuple

from torch.utils.data import Dataset

from artificial_anomalies import disk_anomaly, sample_position
from utils import load_nii_nn, load_files_to_ram, volume_viewer


class BrainDataset(Dataset):
    def __init__(self, files, img_size=None, slice_range: Tuple[int, int] = None,
                 verbose=False):
        super().__init__()

        self.verbose = verbose
        self.img_size = img_size
        self.slice_range = slice_range

        # Function for loading images
        load_fn = partial(load_nii_nn, size=img_size)
        load_fn = partial(load_fn, slice_range=slice_range)
        self.load_fn = load_fn

        # Load data to RAM
        self.samples = self.load_to_ram(files)

    def __len__(self):
        return len(self.samples)

    def print(self, msg):
        if self.verbose:
            print(msg)

    @abc.abstractmethod
    def load_to_ram(self, files):
        pass

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass


class TrainDataset(BrainDataset):
    def load_to_ram(self, files):
        t_start = perf_counter()
        samples = load_files_to_ram(files, self.load_fn)
        samples = [s[None] for vol in samples for s in vol]
        self.print(f"Loaded data in {perf_counter() - t_start:.2f}s")
        return samples

    def __getitem__(self, idx):
        return self.samples[idx]


class TestDataset(BrainDataset):
    def load_to_ram(self, files):
        t_start = perf_counter()
        # Load files
        samples = load_files_to_ram(files, self.load_fn)
        # Flatten list
        samples = [s[None] for vol in samples for s in vol]
        # Create anomalies
        normal, anomal, labels = self.create_anomalies(samples)
        self.print(f"Loaded data in {perf_counter() - t_start:.2f}s")
        return normal, anomal, labels

    def create_anomalies(self, samples):
        anomal_samples = []
        labels = []
        radius = 20
        for img in samples:
            img = img[0]
            position = sample_position(img)
            intensity = random.uniform(0, 1)
            img_anomal, label = disk_anomaly(img, position, radius, intensity)
            anomal_samples.append(img_anomal[None])
            labels.append(label[None])
        return samples, anomal_samples, labels

    def __getitem__(self, idx):
        normal = self.samples[0][idx]
        anomal = self.samples[1][idx]
        label = self.samples[2][idx]
        return normal, anomal, label


if __name__ == "__main__":
    files = glob("/home/felix/datasets/MOOD/brain/train/*.nii.gz")
    # ds = TrainDataset(files[:50], slice_range=(120, 140), verbose=True)
    # x = next(iter(ds))
    # print(x.shape)
    ds = TestDataset(files[:50], slice_range=(120, 140), verbose=True)
    normal, anomal, label = next(iter(ds))
    print(normal.shape, anomal.shape, label.shape)
    import IPython; IPython.embed(); exit(1)
