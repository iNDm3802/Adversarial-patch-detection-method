import numpy as np
from numpy.linalg import inv, pinv
from scipy import stats
import re


def particular_compute_mahalanobis(img, meandiff, output, pos):
    '''
    Computes Mahalanobis distance from current pixel to image as a class

    :param img: source image
    :param meandiff: difference between each pixel and mean value
    :param output: multiplication of inverted covariance matrix and meandiff
    :param pos: position of pixel
    :return: Mahalanobis distance
    '''
    ein_sum = 0
    x = pos[0] * img.shape[0] + pos[1]
    for i in range(len(output[x])):
        ein_sum += output[x][i] * meandiff[x][i]
    return np.sqrt(ein_sum)


def particular_compute_nearest_neighbor(img, pos):
    '''
    Computes distance from current pixel to nearest neighbors

    :param img: source image
    :param pos: position of pixel
    :return: distance to nearest neighbors
    '''
    if len(img.shape) == 3:
        img_transposed = img.transpose(2, 0, 1)
        n = (particular_compute_nearest_neighbor(img_transposed[0], pos)
            + particular_compute_nearest_neighbor(img_transposed[1], pos)
            + particular_compute_nearest_neighbor(img_transposed[2], pos))
        return n
    else:
        area = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                area.append(img[min(max(0, pos[0] + i), img.shape[0] - 1)][min(max(0, pos[1] + j), img.shape[1] - 1)])
        n = (img[pos[0]][pos[1]] - (sum(area) - img[pos[0]][pos[1]]) / (len(area) - 1)) / 255

        return abs(n)


def particular_compute_z(img, pos, mean, stddev):
    '''
    Computes Z-score for current pixel
    For grayscale images only

    :param img: source image
    :param pos: position of pixel
    :param mean: mean pixel value
    :param stddev: pixels' standard deviation
    :return: Z-score
    '''
    if len(img.shape) == 3:
        raise BaseException('Z-score can not be used for colored images')
    return abs((img[pos[0]][pos[1]] - mean) / stddev)


def compute_mahalanobis(img):
    '''
    Computes Mahalanobis distance from each pixel to image as a class

    :param img: source image
    :return: Mahalanobis distance matrix
    '''
    if len(img.shape) == 3:
        arr = np.reshape(img, (img.shape[0] * img.shape[1], 3))

        try:
            invcovar = inv(np.cov(np.transpose(arr)))
        except BaseException:
            invcovar = pinv(np.cov(np.transpose(arr)))
        meandiff = arr - np.mean(arr, axis=0)
        output = np.dot(meandiff, invcovar)

        return np.sqrt(np.einsum('ij,ij->i', output, meandiff)).reshape(img.shape[:-1])
    else:
        arr = np.reshape(img, (img.shape[0] * img.shape[1], 1))

        try:
            invcovar = np.cov(np.transpose(arr)) ** (-1)
        except BaseException:
            invcovar = 0
        meandiff = arr - np.mean(arr, axis=0)
        output = np.dot(meandiff, invcovar)

        return np.sqrt(np.einsum('ij,ij->i', output, meandiff)).reshape(img.shape)


def compute_nearest_neighbor(img):
    '''
    Computes distance from each pixel to nearest neighbors

    :param img: source image
    :return: matrix of distances to nearest neighbors
    '''
    if len(img.shape) == 3:
        img_transposed = img.transpose(2, 0, 1)
        n = (compute_nearest_neighbor(img_transposed[0])
            + compute_nearest_neighbor(img_transposed[1])
            + compute_nearest_neighbor(img_transposed[2]))
        return n
    else:
        res = []
        for row in range(0, img.shape[0]):
            res_row = []
            for col in range(0, img.shape[1]):
                area = []
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        area.append(img[min(max(0, row + i), img.shape[0] - 1)][
                                        min(max(0, col + j), img.shape[1] - 1)])
                res_row.append((img[row][col] - (sum(area) - img[row][col]) / (len(area) - 1)) / 255)
            res.append(res_row)

        res = np.array(res)

        return abs(res)


def compute_z(img):
    '''
    Computes Z-score for each pixel
    For grayscale images only

    :param img: source image
    :return: Z-score matrix
    '''
    if len(img.shape) == 3:
        raise BaseException('Z-score can not be used for colored images')
    return abs(stats.zscore(img, axis=None))


def compute_score(img, mode='mn'):
    '''
    Computes anomaly level scores for each pixel

    :param img: source image
    :param mode: metrics used
        m - Mahalanobis distance
        n - distance to nearest neighbors
        z - z-score
    :return: anomaly level scores matrix
    '''
    pattern = re.compile('^[mnz]{1,3}$')
    if not re.fullmatch(pattern, mode):
        raise BaseException(f'Invalid mode: {mode}')

    score = np.ones((img.shape[0], img.shape[1]))
    score = score.astype(float)

    if 'm' in mode:
        m = compute_mahalanobis(img)
        score *= m
    if 'n' in mode:
        n = compute_nearest_neighbor(img)
        score *= n
    if 'z' in mode:
        z = compute_z(img)
        score *= z

    return score
