import matplotlib.pyplot as plt
import numpy as np


def match_ex4_to_ex1():
    """
    find the closest match between the AE results of experiment 4 and the blur
    results of experiment 1.
    """
    model_type = 'vq-vae'  # Select 'ae' or 'vq-vae'
    ex4_res_path = f'./results/experiment4_full_normal-rec_{model_type}_best_aps.npy'
    ex1_res_path = './results/experiment1_full_aps.npy'

    ex4_res = np.load(ex4_res_path)
    ex1_res = np.load(ex1_res_path)

    min_err = float('inf')
    for i in range(len(ex4_res)):
        err = np.abs(ex4_res - ex1_res[:, i]).sum()
        print(err)
        if err < min_err:
            min_err = err
            min_err_idx = i
    print(f"Min reconstruction error: {min_err} at idx {min_err_idx}")

    blur_match = np.linspace(0, 5, num=len(ex4_res))[min_err_idx]
    matched_curve = ex1_res[:, min_err_idx]

    plt.plot(ex4_res, label=model_type)
    plt.plot(matched_curve, label=f'blur sigma {blur_match}')
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


if __name__ == '__main__':
    match_ex4_to_ex1()
    # match_ex5_to_ex2()
