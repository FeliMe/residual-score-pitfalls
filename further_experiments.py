import matplotlib.pyplot as plt
import numpy as np

from utils import load_nii, blur_img


def match_ex4_to_ex1():
    """
    find the closest match between the AE results of experiment 4 and the blur
    results of experiment 1.
    """
    ex4_lat128 = np.load('./results/experiment4/experiment4_full_normal-rec_ae_lat128_best_aps.npy')
    ex4_lat64 = np.load('./results/experiment4/experiment4_full_normal-rec_ae_lat64_best_aps.npy')
    ex4_lat32 = np.load('./results/experiment4/experiment4_full_normal-rec_ae_lat32_best_aps.npy')
    ex4_vqvae = np.load('./results/experiment4/experiment4_full_normal-rec_vq-vae_best_aps.npy')
    ex1_res = np.load('./results/experiment1/experiment1_full_aps.npy')

    # Match AE to experiment 1
    min_err = float('inf')
    for i in range(len(ex4_lat128)):
        err = np.abs(ex4_lat128 - ex1_res[:, i]).sum()
        print(err)
        if err < min_err:
            min_err = err
            min_err_idx = i
    print(f"Min ap difference AE: {min_err} at idx {min_err_idx}")
    blur_match_ae = np.linspace(0, 5, num=len(ex4_lat128))[min_err_idx]
    matched_curve_ae = ex1_res[:, min_err_idx]

    # Match VQ-VAE to experiment 1
    min_err = float('inf')
    for i in range(len(ex4_vqvae)):
        err = np.abs(ex4_vqvae - ex1_res[:, i]).sum()
        print(err)
        if err < min_err:
            min_err = err
            min_err_idx = i
    print(f"Min ap difference VQ-VAE: {min_err} at idx {min_err_idx}")
    blur_match_vqvae = np.linspace(0, 5, num=len(ex4_vqvae))[min_err_idx]
    matched_curve_vqvae = ex1_res[:, min_err_idx]

    intensities = np.linspace(0, 1, num=len(ex4_lat128))
    plt.plot(intensities, ex4_lat128, label='AE latent dim 128')
    plt.plot(intensities, ex4_lat64, label='AE latent dim 64')
    plt.plot(intensities, ex4_lat32, label='AE latent dim 32')
    plt.plot(intensities, ex4_vqvae, label='VQ-VAE')
    plt.plot(intensities, matched_curve_ae, '--', label=f'Gaussian blur σ={blur_match_ae:.2f}')
    plt.plot(intensities, matched_curve_vqvae, '--', label=f'Gaussian blur σ={blur_match_vqvae:.2f}')
    plt.legend()
    plt.xlabel('intensity')
    plt.ylabel('ap')
    plt.ylim(0, 1)
    plt.show()


def match_ex5_to_ex2():
    """
    find the closest match between the AE results of experiment 5 and the blur
    results of experiment 2.
    """
    model_type = 'vq-vae'  # Select 'ae' or 'vq-vae'
    ex5_res_path = f'./results/experiment5_full_normal-rec_{model_type}_best_intensity06_aps.npy'
    ex2_res_path = './results/experiment2_full_aps_intensity06.npy'

    ex5_res = np.load(ex5_res_path)
    ex2_res = np.load(ex2_res_path)

    min_err = float('inf')
    for i in range(len(ex5_res)):
        err = np.abs(ex5_res - ex2_res[:, i]).sum()
        print(err)
        if err < min_err:
            min_err = err
            min_err_idx = i
    print(f"Min reconstruction error: {min_err} at idx {min_err_idx}")

    blur_match = np.linspace(0, 5, num=len(ex5_res))[min_err_idx]
    matched_curve = ex2_res[:, min_err_idx]

    plt.plot(ex5_res, label=model_type)
    plt.plot(matched_curve, label=f'blur sigma {blur_match}')
    plt.legend()
    plt.xlabel('intensity')
    plt.ylabel('ap')
    plt.ylim(0, 1)
    plt.show()
    import IPython; IPython.embed(); exit(1)


def reconstruction_error_blur():
    img_path = "/home/felix/datasets/MOOD/brain/test_raw/00529.nii.gz"
    volume, _ = load_nii(img_path, primary_axis=2)
    img = volume[volume.shape[0] // 2]

    rec_errs = []

    blurrings = np.linspace(0, 5, num=100)
    for blur in blurrings:
        blurred_img = blur_img(img, blur)
        rec_error = np.abs(img - blurred_img).mean()
        rec_errs.append(rec_error)

    rec_errs = np.array(rec_errs)
    plt.plot(blurrings, rec_errs)
    plt.vlines(0.25, 0, rec_errs.max(), linestyles='dashed', colors='r')
    plt.xlabel('σ')
    plt.ylabel('reconstruction error')
    plt.show()


if __name__ == '__main__':
    # match_ex4_to_ex1()
    # match_ex5_to_ex2()
    reconstruction_error_blur()
