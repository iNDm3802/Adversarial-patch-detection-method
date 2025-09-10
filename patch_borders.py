import cv2
from os import listdir
import numpy as np

from utils import *


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


def parse_stat(path: str):
    file = open(path, 'r', encoding='UTF-8')
    text = file.read()
    file.close()

    data = {}
    lines = text.split('\n')
    while '' in lines:
        lines.remove('')
    for line in lines[1:]:
        l = line.split(', ')
        name = l[0]
        w = int(l[1])
        h = int(l[2])
        y_min = int(l[3])
        y_max = int(l[4]) - 1
        x_min = int(l[5])
        x_max = int(l[6]) - 1

        data[name] = {'width': w, 'height': h,
                      'x_min': x_min, 'x_max': x_max,
                      'y_min': y_min, 'y_max': y_max}

    return data


def find_borders(score, cutoff_l, cutoff_c):
    borders = []

    lines = np.sum(score, axis=0)
    cols = np.sum(score, axis=1)

    compare_lines_mask = lines >= cutoff_l
    compare_cols_mask = cols >= cutoff_c

    if np.max(compare_lines_mask) == False or np.max(compare_cols_mask) == False:
        return False

    borders.append([np.where(compare_cols_mask == True)[0][0], np.where(compare_lines_mask == True)[0][0]])
    borders.append([np.where(compare_cols_mask == True)[0][0], np.where(compare_lines_mask == True)[0][-1]])
    borders.append([np.where(compare_cols_mask == True)[0][-1], np.where(compare_lines_mask == True)[0][0]])
    borders.append([np.where(compare_cols_mask == True)[0][-1], np.where(compare_lines_mask == True)[0][-1]])

    return borders


def check_patch(borders, score):
    area = []
    for i in range(score.shape[0]):
        area.append([0] * score.shape[1])
    for i in range(len(area)):
        for j in range(len(area[i])):
            if borders[0][0] <= i <= borders[2][0] and borders[0][1] <= j <= borders[1][1]:
                area[i][j] = 1
    return area


def evaluate_area(score, area, stat):
    metrics = {'TP': 0, 'FP': 0, 'FN': 0}
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):

            if (area[i][j] == 1
                    and i in range(stat['y_min'], stat['y_max'] + 1)
                    and j in range(stat['x_min'], stat['x_max'] + 1)):
                metrics['TP'] += 1

            elif (area[i][j] == 1
                  and (i not in range(stat['y_min'], stat['y_max'] + 1)
                       or j not in range(stat['x_min'], stat['x_max'] + 1))):
                metrics['FP'] += 1

            elif (area[i][j] == 0
                  and i in range(stat['y_min'], stat['y_max'] + 1)
                  and j in range(stat['x_min'], stat['x_max'] + 1)):
                metrics['FN'] += 1

    return metrics


def compute_metrices(m):
    metrices = {'TP': 0, 'FP': 0, 'FN': 0}

    for item in m:
        metrices['TP'] += item['TP']
        metrices['FP'] += item['FP']
        metrices['FN'] += item['FN']

    metrices['Recall'] = compute_recall(metrices['TP'], metrices['FN'])
    metrices['Precision'] = compute_precision(metrices['TP'], metrices['FP'])
    metrices['F'] = compute_F(recall=metrices['Recall'], precision=metrices['Precision'])

    return metrices


def process_dir(directory: str, stat_file, cutoff: float,
                mean_flag: bool = True, sigma_flag: bool = False, q_flag: bool = False):
    files = [f for f in listdir(directory) if 'txt' not in f and 'original' not in f]
    stat = parse_stat(path=directory + '/' + stat_file)

    m = []

    for f in files:
        score = normalize(parse_score(f"{directory}/{f}"))
        lines = np.sum(score, axis=0)
        cols = np.sum(score, axis=1)

        if mean_flag:
            m_l = np.mean(lines)
            m_c = np.mean(cols)
            # print(f, m_l, m_c)
            cutoff_l = m_l * cutoff
            cutoff_c = m_c * cutoff

            if np.isnan(cutoff_l) or np.isnan(cutoff_c):
                continue

            borders = find_borders(score=score, cutoff_c=cutoff_c, cutoff_l=cutoff_l)

        elif sigma_flag:
            m_l = np.mean(lines)
            m_c = np.mean(cols)
            std_l = np.std(lines)
            std_c = np.std(cols)
            cutoff_l = m_l + cutoff * std_l
            cutoff_c = m_c + cutoff * std_c

            borders = find_borders(score=score, cutoff_c=cutoff_c, cutoff_l=cutoff_l)

        elif q_flag:
            max_l = np.max(lines)
            q75_l = np.quantile(lines, 0.75)
            q25_l = np.quantile(lines, 0.25)
            try:
                cutoff_l = (max_l - q75_l) / (q75_l - q25_l)
            except ZeroDivisionError:
                cutoff_l = 0

            max_c = np.max(cols)
            q75_c = np.quantile(cols, 0.75)
            q25_c = np.quantile(cols, 0.25)
            try:
                cutoff_c = (max_c - q75_c) / (q75_c - q25_c)
            except ZeroDivisionError:
                cutoff_c = 0

            borders = find_borders(score=score, cutoff_c=cutoff_c, cutoff_l=cutoff_l)

        else:
            borders = find_borders(score=score, cutoff_c=cutoff, cutoff_l=cutoff)

        name = f.split(',')[0]
        name = name.split("'")[1]
        name += '.png'

        s = stat[name]
        if not borders:
            m.append({'TP': 0, 'FP': 0, 'FN': s['width'] * s['height']})
            continue
        area = check_patch(borders=borders, score=score)

        m.append(evaluate_area(score=score, area=area, stat=s))

    metrics = compute_metrices(m)

    # print(metrics)
    return metrics


dirs = []
dirs.append('E:/_scores/CIFAR10/lavan')
# dirs.append('E:/_scores/ImageNet/lavan')
stat_file = 'images_stat_cifar10.txt'

min_cutoff = 0
max_cutoff = 10
step_cutoff = 0.1

mean = False # 2-10
sigma = False # 2-10
q = True # 2-10

suff = 'abs'
if mean:
    suff = 'mean'
elif sigma:
    suff = 'sigma'
elif q:
    suff = 'q'

cutoff = min_cutoff
metrices = {}

f = open(f"_!_patch_{suff}.txt", 'w')
head = "Cutoff\tTP\tFP\tFN\tPrecision\tRecall\tF1\n"
f.write(head)
f.close()
while cutoff <= max_cutoff:
    metrics = process_dir(directory=dirs[0], stat_file=stat_file, cutoff=cutoff,
                           mean_flag=mean, sigma_flag=sigma, q_flag=q)
    metrices[cutoff] = metrics

    f = open(f"_!_patch_{suff}.txt", 'a')
    text = f"{cutoff}\t"
    text += f"{metrices[cutoff]['TP']}\t{metrices[cutoff]['FP']}\t{metrices[cutoff]['FN']}\t"
    text += f"{round(metrices[cutoff]['Precision'], 4)}\t"
    text += f"{round(metrices[cutoff]['Recall'], 4)}\t"
    text += f"{round(metrices[cutoff]['F'], 4)}\n"
    f.write(text)
    f.close()

    print(f"{cutoff} - {metrices[cutoff]['F']}")
    print(metrices[cutoff])

    cutoff += step_cutoff

raise BaseException("FINISHED")
