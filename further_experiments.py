import matplotlib.pyplot as plt
import numpy as np

from utils import load_nii, blur_img


def match_ex4_to_ex1():
    """
    find the closest match between the AE results of experiment 4 and the blur
    results of experiment 1.
    """
    ex4_lat128 = np.load('./results/experiment4/experiment4_full_normal-rec_ae_lat128_best_aps.npy')
    # ex4_lat64 = np.load('./results/experiment4/experiment4_full_normal-rec_ae_lat64_best_aps.npy')
    ex4_lat32 = np.load('./results/experiment4/experiment4_full_normal-rec_ae_lat32_best_aps.npy')
    spatial1 = np.load('./results/experiment4/experiment4_full_normal-rec_spatial-ae_lat1_best_aps.npy')
    spatial2 = np.load('./results/experiment4/experiment4_full_normal-rec_spatial-ae_lat2_best_aps.npy')
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
    plt.plot(intensities, ex4_lat128, label='AE latent 128')
    # plt.plot(intensities, ex4_lat64, label='AE latent 64')
    plt.plot(intensities, ex4_lat32, label='AE latent 32')
    plt.plot(intensities, spatial1, label='Spatial-AE latent 8x8x1')
    plt.plot(intensities, spatial2, label='Spatial-AE latent 8x8x2')
    plt.plot(intensities, ex4_vqvae, label='VQ-VAE')
    plt.plot(intensities, matched_curve_ae, '--', label=f'Gaussian blur σ={blur_match_ae:.2f}')
    plt.plot(intensities, matched_curve_vqvae, '--', label=f'Gaussian blur σ={blur_match_vqvae:.2f}')
    plt.legend(loc=4)
    plt.xlabel('intensity')
    plt.ylabel('AP')
    plt.ylim(0, 1)
    plt.show()


def match_ex3_to_ex1():
    """
    find the closest match between the results of experiment 3 and the blur
    results of experiment 1.
    """
    pixel_shuffle = np.load('./results/experiment3/experiment3_full_pixel_shuffle_aps.npy')
    sink_deform = np.load('./results/experiment3/experiment3_full_sink_deformation_aps.npy')
    source_deform = np.load('./results/experiment3/experiment3_full_source_deformation_aps.npy')
    ex1_res = np.load('./results/experiment1/experiment1_full_aps.npy')

    min_err = float('inf')
    for i in range(len(pixel_shuffle)):
        err = np.abs(pixel_shuffle - ex1_res[i]).sum()
        print(err)
        if err < min_err:
            min_err = err
            min_err_idx = i
    print(f"Min reconstruction error: {min_err} at idx {min_err_idx}")

    intensity_match = np.linspace(0, 1, num=len(pixel_shuffle))[min_err_idx]
    matched_curve = ex1_res[min_err_idx]

    blurrings = np.linspace(0, 5, num=100)
    plt.plot(blurrings, sink_deform, label='Sink deformation')
    plt.plot(blurrings, source_deform, label='Source deformation')
    plt.plot(blurrings, pixel_shuffle, label='Pixel shuffle')
    plt.plot(blurrings, matched_curve, label=f'Experiment 1: intensity {intensity_match:.2f}')
    plt.vlines(0.25, 0, 1, linestyles='dashed', colors='r')
    plt.legend()
    plt.xlabel('σ')
    plt.ylabel('AP')
    plt.ylim(0, 1)
    plt.show()
    import IPython; IPython.embed(); exit(1)


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


def plot_experiment4_1():
    lat128 = np.load('./results/experiment4/experiment4_full_anomal-rec_ae_lat128_best_aps.npy')
    # lat64 = np.load('./results/experiment4/experiment4_full_anomal-rec_ae_lat64_best_aps.npy')
    lat32 = np.load('./results/experiment4/experiment4_full_anomal-rec_ae_lat32_best_aps.npy')
    spatial1 = np.load('./results/experiment4/experiment4_full_anomal-rec_spatial-ae_lat1_best_aps.npy')
    spatial2 = np.load('./results/experiment4/experiment4_full_anomal-rec_spatial-ae_lat2_best_aps.npy')
    vqvae = np.load('./results/experiment4/experiment4_full_anomal-rec_vq-vae_best_aps.npy')

    intensities = np.linspace(0, 1, num=len(lat128))
    plt.plot(intensities, lat128, label='AE latent 128')
    # plt.plot(intensities, lat64, label='AE latent 64')
    plt.plot(intensities, lat32, label='AE latent 32')
    plt.plot(intensities, spatial1, label='Spatial-AE latent 8x8x1')
    plt.plot(intensities, spatial2, label='Spatial-AE latent 8x8x2')
    plt.plot(intensities, vqvae, label='VQ-VAE')
    plt.legend()
    plt.xlabel('intensity')
    plt.ylabel('AP')
    plt.ylim(0, 1)
    plt.show()


def plot_rec_err_vs_ap():
    plt.plot(0.03465, 0.262, 'o', label='AE latent 128')
    plt.plot(0.03725, 0.294, 'o', label='AE latent 64')
    plt.plot(0.03748, 0.320, 'o', label='AE latent 32')
    plt.plot(0.03637, 0.312, 'o', label='Spatial-AE latent 8x8x1')
    plt.plot(0.03249, 0.273, 'o', label='Spatial-AE latent 8x8x2')
    # plt.plot(0.01052, 0.067, 'o', label='Skip-AE')
    # plt.plot(0.01093, 0.375, 'o', label='VQ-VAE')

    plt.xlabel('AP')
    plt.ylabel('reconstruction error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # match_ex4_to_ex1()
    # match_ex3_to_ex1()
    # reconstruction_error_blur()
    # plot_experiment4_1()
    plot_rec_err_vs_ap()
