from datetime import datetime
from functools import partial
from glob import glob
import os
from typing import List, Tuple, Sequence, Callable

from matplotlib import cm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_hist
from skimage.transform import resize
from sklearn.metrics import average_precision_score


DATAROOT = os.environ.get("DATAROOT")
if DATAROOT is None:
    raise EnvironmentError("Set the $DATAROOT environment variable to your data directory")
MOODROOT = os.path.join(DATAROOT, "MOOD/brain")


def load_files_to_ram(files: Sequence, load_fn: Callable,
                      num_processes: int = 48):
    pool = Pool(num_processes)
    results = []

    results = pool.map(load_fn, files)

    pool.close()
    pool.join()

    return results


def show(imgs: List[np.ndarray], seg: List[np.ndarray] = None,
         path: str = None) -> None:

    if not isinstance(imgs, list):
        imgs = [imgs]
    n = len(imgs)
    fig = plt.figure()

    for i in range(n):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(imgs[i], cmap="gray", vmin=0., vmax=1.)

        if path is not None:
            plt.axis('off')

        if seg is not None:
            plt.imshow(seg[i], cmap="jet", alpha=0.3)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_landscape(X, Y, Z, ax_labels: Tuple[str, str, str] = None,
                   path: str = None) -> None:
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X_, Y_ = np.meshgrid(X, Y)

    # Plot the surface.
    ax.plot_surface(X_, Y_, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=True)

    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_zlabel(ax_labels[2])

    # Limit z-axis
    ax.set_zlim(0., 1.)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_heatmap(x, y, z, ax_labels: Tuple[str, str] = None, path: str = None):
    # Init figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot heatmap
    extent = [x.min(), x.max(), y.min(), y.max()]
    im = ax.imshow(np.flip(z, axis=0), cmap=cm.coolwarm, interpolation='nearest', extent=extent)

    # Add colorbar
    fig.colorbar(im)

    # Set axis labels
    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])

    # Set aspect ratio
    ax.set_aspect('equal')
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])))

    plt.vlines(0.25, 0, 1, linestyles='dashed')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_curve(x: np.ndarray, y: np.ndarray, ax_titles: Tuple[str, str] = None,
               path: str = None) -> None:
    plt.plot(x, y)

    plt.ylim(0., 1.)

    if ax_titles is not None:
        plt.xlabel(ax_titles[0])
        plt.ylabel(ax_titles[1])

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def load_nii(path: str, size: int = None, primary_axis: int = 0,
             dtype: str = "float32"):
    """Load a neuroimaging file with nibabel, [w, h, slices]
    https://nipy.org/nibabel/reference/nibabel.html
    Args:
        path (str): Path to nii file
        size (int): Optional. Output size for h and w. Only supports rectangles
        primary_axis (int): Primary axis (the one to slice along, usually 2)
        dtype (str): Numpy datatype
    Returns:
        volume (np.ndarray): Of shape [w, h, slices]
        affine (np.ndarray): Affine coordinates (rotation and translation),
                             shape [4, 4]
    """
    # Load file
    data = nib.load(path, keep_file_open=False)
    volume = data.get_fdata(caching='unchanged')  # [w, h, slices]
    affine = data.affine

    # Squeeze optional 4th dimension
    if volume.ndim == 4:
        volume = volume.squeeze(-1)

    # Resize if size is given and if necessary
    if size is not None and (volume.shape[0] != size or volume.shape[1] != size):
        volume = resize(volume, [size, size, size])

    # Convert
    volume = volume.astype(np.dtype(dtype))

    # Move primary axis to first dimension
    volume = np.moveaxis(volume, primary_axis, 0)

    return volume, affine


def load_nii_nn(path: str, size: int = None,
                slice_range: Tuple[int, int] = None,
                dtype: str = "float32"):
    vol = load_nii(path, size, primary_axis=2, dtype=dtype)[0]

    if slice_range is not None:
        vol = vol[slice_range[0]:slice_range[1]]

    return vol


def histogram_equalization(img):
    # Create equalization mask
    mask = np.where(img > 0, 1, 0)
    # Equalize
    img = equalize_hist(img, nbins=256, mask=mask)
    # Assure that background still is 0
    img *= mask

    return img


def average_precision(target: np.ndarray, pred: np.ndarray) -> float:
    return average_precision_score(target.reshape(-1), pred.reshape(-1))


def blur_img(img: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(img, sigma=sigma)


def get_training_timings(start_time, current_step, num_steps):
    time_elapsed = datetime.now() - datetime.fromtimestamp(start_time)
    # self.current_epoch starts at 0
    time_per_step = time_elapsed / current_step
    time_left = (num_steps - current_step) * time_per_step
    return time_elapsed, time_per_step, time_left


def load_mood_test_data() -> np.ndarray:
    """
    Loads the test data for the MOOD challenge.
    :return: The test data as a numpy array.
    """
    files = glob(os.path.join(MOODROOT, 'test_raw/*.nii.gz'))
    load_fn = partial(load_nii_nn, slice_range=(128, 129))
    data = load_files_to_ram(files, load_fn)
    data = np.stack([s for vol in data for s in vol], axis=0)
    return data
