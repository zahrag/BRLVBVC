import numpy as np


def normalize(X, axis=0):
    """
    Normalizes an array but takes into account zero elements
    :param X: 1D and 2D tested, nD should work
    :return: Normalized array
    """
    mask = X > 0
    if len(X.shape) == 1:
        # Handle 1D-numpy array
        X[mask] /= np.linalg.norm(X, ord=1, axis=axis)
    else:
        # Handle 2D-numpy arrays
        n = np.linalg.norm(X, ord=1, axis=axis)
        n[n == 0] = 1.  # Set 0 norm to 1
        X /= np.expand_dims(n, axis=axis)

    return X


def get_margins():
    # Make Perceptual State Boundaries for all Learners
    hei_min = 0
    hei_max = 1
    wid_min = 0
    wid_max = 1

    hei_1 = 0.4
    wid_1 = 0.3
    wid_2 = 0.7

    image_margin = []
    image_margin.append(np.array(([hei_min, hei_1, wid_min, wid_1])))
    image_margin.append(np.array(([hei_min, hei_1, wid_1, wid_2])))
    image_margin.append(np.array(([hei_min, hei_1, wid_2, wid_max])))
    image_margin.append(np.array(([hei_1, hei_max, wid_min, wid_1])))
    image_margin.append(np.array(([hei_1, hei_max, wid_1, wid_2])))
    image_margin.append(np.array(([hei_1, hei_max, wid_2, wid_max])))

    return image_margin

