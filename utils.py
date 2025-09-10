from math import sqrt


def amplify(image, mask):
    '''
    Applies mask on image

    :param image: source image
    :param mask: positions of malicious pixels
    :return: image with only malicious pixels
    '''
    img = image

    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if mask[i][j] == 0:
                for c in range(0, 3):
                    img[i][j][c] = 0
            else:
                for c in range(0, 3):
                    img[i][j][c] = image[i][j][c]

    return img


def normalize(a):
    '''
    Normalizes 2D-array

    :param a: array
    :return: normalized array
    '''
    a_min = a.min(axis=(0, 1), keepdims=True)
    a_max = a.max(axis=(0, 1), keepdims=True)
    return (a - a_min) / (a_max - a_min)


def compute_accuracy(TP: int, FP: int, TN: int, FN: int):
    '''
    Computes accuracy

    :param TP: number of true positives
    :param FP: number of false positives
    :param TN: number of true negatives
    :param FN: number of false negatives
    :return: accuracy
    '''
    try:
        return (TP + TN) / (TP + FP + TN + FN)
    except BaseException:
        return 0


def compute_precision(TP: int, FP: int):
    '''
    Computes precision

    :param TP: number of true positives
    :param FP: number of false positives
    :return: precision
    '''
    try:
        return TP / (TP + FP)
    except BaseException:
        return 0


def compute_recall(TP: int, FN: int):
    '''
    Computes recall

    :param TP: number of true positives
    :param FN: number of false negatives
    :return: recall
    '''
    try:
        return TP / (TP + FN)
    except BaseException:
        return 0


def compute_F(precision: float, recall: float, beta: float = 1):
    '''
    Computes F-score

    :param precision: precision
    :param recall: recall
    :param beta: weight coefficient
    :return: F-score
    '''
    try:
        return (1 + pow(beta, 2)) * precision * recall / (pow(beta, 2) * precision + recall)
    except BaseException:
        return 0


def compute_MCC(TP: int, FP: int, TN: int, FN: int):
    '''
    Computes Mattheus correlation coefficient

    :param TP: number of true positives
    :param FP: number of false positives
    :param TN: number of true negatives
    :param FN: number of false negatives
    :return: Mattheus correlation coefficient
    '''
    try:
        return (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except BaseException:
        return -1
