from preprocessing import compute_score, particular_compute_mahalanobis, particular_compute_nearest_neighbor, \
    particular_compute_z
from utils import normalize
import numpy as np
from numpy.linalg import inv, pinv
import re


def detect_perturbation(img, cutoff: float = 1, mode='mn', is_normalized=True):
    '''
    Detects perturbation pixels

    :param img: source image
    :param cutoff: minimum value of anomaly level
    :param mode: metrics used
        m - Mahalanobis distance
        n - distance to nearest neighbors
        z - z-score
    :param is_normalized: is score matrix normalized
    :return: perturbation pixels
    '''
    score = compute_score(img, mode=mode)
    if is_normalized:
        score = normalize(score)

    perturbation = []
    for i in range(0, score.shape[0]):
        for j in range(0, score.shape[1]):
            if score[i][j] >= cutoff:
                perturbation.append({'x': i, 'y': j, 'score': score[i][j]})

    return perturbation


def detect_attack(img, cutoff: float = 0, mode='mn'):
    '''
    Checks if an image was attacked

    :param img: source image
    :param cutoff: minimum value of anomaly level
    :param mode: metrics used
        m - Mahalanobis distance
        n - distance to nearest neighbors
        z - z-score
    :return: attack flag
    '''
    pattern = re.compile('^[mnz]{1,3}$')
    if not re.fullmatch(pattern, mode):
        raise BaseException(f'Invalid mode: {mode}')

    if 'm' in mode:
        if len(img.shape) == 3:
            arr = np.reshape(img, (img.shape[0] * img.shape[1], 3))
            try:
                invcovar = inv(np.cov(np.transpose(arr)))
            except BaseException:
                invcovar = pinv(np.cov(np.transpose(arr)))

        else:
            arr = np.reshape(img, (img.shape[0] * img.shape[1], 1))
            try:
                invcovar = np.cov(np.transpose(arr)) ** (-1)
            except BaseException:
                invcovar = 0
        meandiff = arr - np.mean(arr, axis=0)
        output = np.dot(meandiff, invcovar)

    if 'n' in mode:
        if len(img.shape) == 3:
            img_transposed = img.transpose(2, 0, 1)
        else:
            img_transposed = None

    if 'z' in mode:
        if len(img.shape) == 3:
            raise BaseException(f'Invalid mode for colored image: {mode}\nZ-score is only for grayscale')
        else:
            mean = img.mean()
            stddev = img.stddev()

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            score = 1

            if 'm' in mode:
                score *= particular_compute_mahalanobis(img, meandiff, output, [row, col])

            if 'n' in mode:
                if img_transposed:
                    score *= (particular_compute_nearest_neighbor(img_transposed[0], [row, col])
                              + particular_compute_nearest_neighbor(img_transposed[1], [row, col])
                              + particular_compute_nearest_neighbor(img_transposed[2], [row, col]))
                else:
                    score *= particular_compute_nearest_neighbor(img, [row, col])

            if 'z' in mode:
                score *= particular_compute_z(img, [row, col], mean, stddev)

            if score > cutoff:
                return True

    return False
