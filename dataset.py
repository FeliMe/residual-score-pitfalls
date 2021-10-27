from functools import partial
from glob import glob
from multiprocessing import Pool
from time import time
from typing import Tuple

from torch.utils.data import Dataset

from utils import load_nii_nn, volume_viewer


def load_files_to_ram(files, load_fn, num_processes=48):
    pool = Pool(num_processes)
    results = []

    results = pool.map(load_fn, files)

    pool.close()
    pool.join()

    return results


class TrainDataset(Dataset):
    def __init__(self, files, img_size=None, slice_range: Tuple[int, int]=None,
                 load_to_ram=True, verbose=False):
        super().__init__()

        self.verbose = verbose
        self.img_size = img_size
        self.slice_range = slice_range
        self.load_to_ram = load_to_ram

        # Function for loading images
        load_fn = partial(load_nii_nn, size=img_size)
        load_fn = partial(load_fn, slice_range=slice_range)
        self.load_fn = load_fn

        if load_to_ram:
            t_start = time()
            samples = load_files_to_ram(files, self.load_fn)
            self.samples = [s for vol in samples for s in vol]
            self.print(f"Loaded data in {time() - t_start:.2f}s")
        else:
            self.samples = files

    def __len__(self):
        return len(self.samples)

    def print(self, msg):
        if self.verbose:
            print(msg)

    def __getitem__(self, idx):
        if self.load_to_ram:
            return self.samples[idx]


if __name__ == "__main__":
    files = glob("/home/felix/datasets/MOOD/brain/train/*.nii.gz")
    ds = TrainDataset(files[:50], slice_range=(120, 140), verbose=True)
    x = next(iter(ds))
    print(x.shape)
    import IPython ; IPython.embed() ; exit(1)
