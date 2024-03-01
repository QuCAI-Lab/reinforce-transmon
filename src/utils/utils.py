import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import os

def min_max_norm(x: np.ndarray) -> np.ndarray:
    '''
    This function computes the Min-Max feature scaling (normalization) for either a vector or a matrix.

    Args:
        - x (numpy.ndarray): state to be normalized.

    Returns:
        - norm_X (numpy.ndarray): the normalized state.
    '''
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)

def create_gif(save_dir, image_dir, gif_name):
    images = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.png'):
            images.append(imageio.imread(os.path.join(image_dir, filename)))
    save_dir = os.path.join(save_dir, gif_name)
    imageio.mimsave(save_dir, images, format='GIF', duration=.2, loop=0)


def plot_learning_curve(scores, save_dir, save_filename="score_trend"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    x = np.arange(1, len(scores) + 1)
    
    plt.plot(x, scores)
    plt.title("Learning Curve")
    plt.xlabel("Training Episodes")
    plt.ylabel("RL Score")
    plt.savefig(os.path.join(save_dir, save_filename+".png"))
    plt.close()
