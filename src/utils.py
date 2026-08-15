import math
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return  betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
        
def betas_for_alpha_bar(num_diffusion_timesteps:int, alpha_bar, max_beta=0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def print_accu_confmat_f1score(Y_true, Y_pred, txt_labels=None):
    act_accu = accuracy_score(y_true=Y_true, y_pred=Y_pred)
    print("***[MY EVALUATION RESULT]*** Accuracy: " + str(round(act_accu, 4) * 100) + "%\n")

    conf_mat = confusion_matrix(y_true=Y_true, y_pred=Y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print("***[RESULT]***  Confusion Matrix:")
    if txt_labels != None:
        print(" | ".join(txt_labels))
    print(np.array(conf_mat).round(3) * 100)
    print()

    f1act = f1_score(y_true=Y_true, y_pred=Y_pred, average=None).mean()
    print("***[RESULT]*** Averaged F-1 Score: " + str(f1act * 100) + "\n")

def print_accu_score(Y_true, Y_pred):
    accu = accuracy_score(y_true=Y_true, y_pred=Y_pred)
    print("    " + str(accu))
    print("    ***[EVALUATION RESULT]*** Accuracy: " + str(round(accu, 4) * 100) + "%\n")

def plot_compare(raw_data, recon_data, n_plots):
    for i in range(n_plots):
        plt.figure(figsize=(150,150)) 
        for j in range(2):
            plt.subplot(2, n_plots, i*2+1)
            plt.imshow((raw_data[i][0]), vmin=-5, vmax=5, cmap='viridis')
            plt.subplot(2, n_plots, (i+1)*2)
            plt.imshow((recon_data[i][0]), vmin=-5, vmax=5, cmap='viridis')
    plt.close()