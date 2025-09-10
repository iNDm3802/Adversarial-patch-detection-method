import cv2
import math
from os import listdir
import numpy as np

from utils import *


def normalize(a):
    a_min = a.min(axis=(0, 1), keepdims=True)
    a_max = a.max(axis=(0, 1), keepdims=True)
    return (a - a_min) / (a_max - a_min)


def parse_score(path: str):
    mmm = path.split('max=')[1].split('.p')[0]
    if ' ' in mmm:
        mmm = mmm.split(' ')[0]
    max = float(mmm)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = normalize(img)
    score *= max
    return score


def dist(a, b):
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    return math.sqrt(dx ** 2 + dy ** 2)


# def check(x: int, y: int):
#     M = {'x': (x - 1) / 2, 'y': (y - 1) / 2}
#     mean_dist = 0
#     for i in range(x):
#         for j in range(y):
#             mean_dist += dist(M, {'x': i, 'y': j})
#     mean_dist /= x * y
#
#     diag = math.sqrt(x ** 2 + y ** 2)
#
#     r = mean_dist / diag
#
#     return r
#
#
# def check_2(R: int):
#     O = {'x': R, 'y': R}
#
#     count = 0
#     mean_dist = 0
#     for i in range(0, 2 * R + 1):
#         for j in range(0, 2 * R + 1):
#             dd = dist(O, {'x': i, 'y': j})
#             if dd <= R:
#                 mean_dist += dd
#                 count += 1
#     mean_dist /= count
#
#     r = mean_dist / R
#
#     return r


def is_patched(score, cutoff, cutoff_dist):
    # square
    susp_pix = []

    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            if score[i][j] >= cutoff:
                susp_pix.append({'x': i, 'y': j, 'score': score[i][j]})

    if len(susp_pix) <= 3:
        return False

    mean_x, mean_y = 0, 0
    for pixel in susp_pix:
        mean_x += pixel['x']
        mean_y += pixel['y']
    mean_x /= len(susp_pix)
    mean_y /= len(susp_pix)
    mean_pix = {'x': mean_x, 'y': mean_y}

    mean_dist = 0
    for pixel in susp_pix:
        mean_dist += dist(pixel, mean_pix)
    mean_dist /= len(susp_pix)

    ctf = math.sqrt(len(susp_pix) / 2)
    if mean_dist <= ctf:
        return True
    return False


def is_patched_dir(directory: str, cutoff_score: float = 1, cutoff_dist: float = 1, write: bool = True):
    metrices = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    files = [f for f in listdir(directory) if 'txt' not in f and 'original' not in f]
    attack = {}
    for f in files:
        score = normalize(parse_score(f"{directory}/{f}"))
        attack[f] = is_patched(score, cutoff_score, cutoff_dist)

    for key in attack.keys():
        if 'adversarial' in key and attack[key]:
            metrices['TP'] += 1
        elif 'adversarial' not in key and attack[key]:
            metrices['FP'] += 1
        elif 'adversarial' in key and not attack[key]:
            metrices['FN'] += 1
        elif 'adversarial' not in key and not attack[key]:
            metrices['TN'] += 1

    if write:
        f = open(f"patch_{directory.split('/')[-2]}_{directory.split('/')[-1]}.txt", 'a')
        text = f"{cutoff_score}\t{cutoff_dist}\t{metrices['TP']}\t{metrices['FP']}\t{metrices['FN']}\t{metrices['TN']}\n"
        f.write(text)
        f.close()

    return metrices


def compute_metrices(m):
    metrices = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    for item in m:
        metrices['TP'] += item['TP']
        metrices['FP'] += item['FP']
        metrices['FN'] += item['FN']
        metrices['TN'] += item['TN']

    metrices['Accuracy'] = compute_accuracy(metrices['TP'], metrices['FP'], metrices['FN'], metrices['TN'])
    metrices['Recall'] = compute_recall(metrices['TP'], metrices['FN'])
    metrices['Precision'] = compute_precision(metrices['TP'], metrices['FP'])
    metrices['F'] = compute_F(recall=metrices['Recall'], precision=metrices['Precision'])

    return metrices


# x_max = 1100
# y_max = 100
#
# rs = []
# for x in range(100, x_max, 100):
#     for y in range(10, y_max, 100):
#         r = check(x, y)
#         rs.append({'x': x, 'y': y, 'r': r})

# R_max = 10
# for x in range(1, R_max + 1, 1):
#     r = check_2(x)
#     rs.append({'R': x, 'r': r})

# rr = []
# for i in rs:
#     rr.append(i['r'])
# rr = np.array(rr)
#
# r_max = rr.max()
# r_min = rr.min()
# r_mean = rr.mean()

raise BaseException("FINISHED")

dirs = []
dirs.append('E:/_scores/CIFAR10/all_lavan')
# dirs.append('E:/_scores/ImageNet/lavan')
dirs.append('E:/_scores/CIFAR10/all_jsma')

min_cutoff = 0.5
max_cutoff = 1
step_cutoff = 10

cutoff = min_cutoff
metrices = {}

# f = open(f"__patch.txt", 'w')
# f = open(f"_.!..patch.txt", 'w')
# head = "Cutoff\tTP\tFP\tFN\tTN\tAccuracy\tPrecision\tRecall\tF1\n"
# f.write(head)
# f.close()
while cutoff <= max_cutoff:
    mmm = []
    for d in dirs:
        mmm.append(is_patched_dir(directory=d, cutoff_score=0.15, cutoff_dist=cutoff))
    metrices[cutoff] = compute_metrices(mmm)

    # f = open(f"__patch.txt", 'a')
    f = open(f"_.!..patch.txt", 'a')
    text = f"{cutoff}\t"
    text += f"{metrices[cutoff]['TP']}\t{metrices[cutoff]['FP']}\t{metrices[cutoff]['FN']}\t{metrices[cutoff]['TN']}"
    text += f"{round(metrices[cutoff]['Accuracy'], 4)}\t"
    text += f"{round(metrices[cutoff]['Precision'], 4)}\t"
    text += f"{round(metrices[cutoff]['Recall'], 4)}\t"
    text += f"{round(metrices[cutoff]['F'], 4)}\n\n\n"
    f.write(text)
    f.close()

    print(metrices[cutoff])
    print(f"{cutoff} - {metrices[cutoff]['F']}")

    cutoff += step_cutoff

raise BaseException("FINISHED")
