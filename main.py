from os import listdir
from os.path import join
import cv2
import time

from detection import *
from utils import *
from LaVan_preprocessing import *


def check_attack(score, cutoff):
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            if score[i][j] >= cutoff:
                return True
    return False


def test_attack_detection(dir, cutoff_min=0, cutoff_max=10, cutoff_step=0.1, mode='mn', save=True):
    files = [f for f in listdir(dir)]
    files = files[:2000]
    num = 0
    evaluation = {}
    for f in files:
        num += 1
        print(f"Processing: {round(num * 100 / len(files), 8)}%")
        print(f)

        img = cv2.imread(join(dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score = compute_score(img, mode=mode)

        cutoff = cutoff_min
        while cutoff <= cutoff_max:
            if cutoff not in evaluation.keys():
                evaluation[cutoff] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

            if 'adversarial' in f:
                if check_attack(score, cutoff):
                    evaluation[cutoff]['TP'] += 1
                else:
                    evaluation[cutoff]['FN'] += 1

            elif 'original' in f:
                if check_attack(score, cutoff):
                    evaluation[cutoff]['FP'] += 1
                else:
                    evaluation[cutoff]['TN'] += 1

            else:
                print(f"Unexpected filename: {f} - no adversity marker")

            cutoff += cutoff_step

    for cutoff in evaluation.keys():
        evaluation[cutoff]['ACC'] = compute_accuracy(TP=evaluation[cutoff]['TP'],
                                                     FP=evaluation[cutoff]['FP'],
                                                     FN=evaluation[cutoff]['FN'],
                                                     TN=evaluation[cutoff]['TN'])
        evaluation[cutoff]['P'] = compute_precision(TP=evaluation[cutoff]['TP'],
                                                    FP=evaluation[cutoff]['FP'])
        evaluation[cutoff]['R'] = compute_recall(TP=evaluation[cutoff]['TP'],
                                                 FN=evaluation[cutoff]['FN'])
        evaluation[cutoff]['F'] = compute_F(precision=evaluation[cutoff]['P'],
                                            recall=evaluation[cutoff]['R'])

    if save:
        text = "cutoff, TP, FP, FN, TN, Accuracy, Precision, Recall, F1-score\n"
        for cutoff in evaluation.keys():
            text += f"{cutoff},{evaluation[cutoff]['TP']},{evaluation[cutoff]['FP']},{evaluation[cutoff]['FN']},{evaluation[cutoff]['TN']},{evaluation[cutoff]['ACC']},{evaluation[cutoff]['P']},{evaluation[cutoff]['R']},{evaluation[cutoff]['F']}\n"

        stat = open("attack_detection_stat.txt", 'w')
        stat.write(text)
        stat.close()

    print("test_attack_detection is finished")
    return evaluation


def test_patch_detection(dir, patch_pointer=(0, 0), patch_side=10, cutoff_min=0, cutoff_max=10, cutoff_step=0.1,
                         mode='mn', is_normalized=False, save=True):
    # patch_pointer - top left angle
    # patch_side - width and height of the patch

    files = [f for f in listdir(dir) if 'adversarial' in f]
    files = files[1:5]
    num = 0
    evaluation = {}
    for f in files:
        num += 1
        print(f"Processing: {round(num * 100 / len(files), 8)}%")
        img = cv2.imread(join(dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score = compute_score(img, mode=mode)
        if is_normalized:
            score = normalize(score)

        cutoff = cutoff_min
        while cutoff <= cutoff_max:
            if cutoff not in evaluation.keys():
                evaluation[cutoff] = {'TP': 0, 'FP': 0, 'FN': 0}

            for i in range(score.shape[0]):
                for j in range(score.shape[1]):

                    if (score[i][j] >= cutoff
                            and i in range(patch_pointer[0], patch_pointer[0] + patch_side)
                            and j in range(patch_pointer[1], patch_pointer[1] + patch_side)):
                        evaluation[cutoff]['TP'] += 1

                    elif (score[i][j] >= cutoff
                          and ((i not in range(patch_pointer[0], patch_pointer[0] + patch_side))
                               or (j not in range(patch_pointer[1], patch_pointer[1] + patch_side)))):
                        evaluation[cutoff]['FP'] += 1

                    elif (score[i][j] < cutoff
                          and i in range(patch_pointer[0], patch_pointer[0] + patch_side)
                          and j in range(patch_pointer[1], patch_pointer[1] + patch_side)):
                        evaluation[cutoff]['FN'] += 1

            cutoff += cutoff_step

    for cutoff in evaluation.keys():
        evaluation[cutoff]['P'] = compute_precision(TP=evaluation[cutoff]['TP'],
                                                    FP=evaluation[cutoff]['FP'])
        evaluation[cutoff]['R'] = compute_recall(TP=evaluation[cutoff]['TP'],
                                                 FN=evaluation[cutoff]['FN'])
        evaluation[cutoff]['F'] = compute_F(precision=evaluation[cutoff]['P'],
                                            recall=evaluation[cutoff]['R'])

    if save:
        text = "cutoff, TP, FP, FN, Precision, Recall, F1-score\n"
        for cutoff in evaluation.keys():
            text += f"{cutoff},{evaluation[cutoff]['TP']},{evaluation[cutoff]['FP']},{evaluation[cutoff]['FN']},{evaluation[cutoff]['P']},{evaluation[cutoff]['R']},{evaluation[cutoff]['F']}\n"

        stat = open("patch_detection_stat.txt", 'w')
        stat.write(text)
        stat.close()

    print("test_patch_detection is finished")
    return evaluation


def test_patch_detection_new(dir, patch_pointer=(0, 0), patch_side=10,
                             cutoff_min=0, cutoff_max=10, cutoff_step=0.1,
                             cutoff_out_min=0, cutoff_out_max=3, cutoff_out_step=0.1,
                             mode='mn', is_normalized=False, save=True):
    # patch_pointer - top left angle
    # patch_side - width and height of the patch

    files = [f for f in listdir(dir) if 'adversarial' in f]
    files = files[1:5]
    num = 0
    evaluation = {}
    for f in files:
        img = cv2.imread(join(dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n = compute_nearest_neighbor(img)

        cutoff_out = cutoff_out_min
        while cutoff_out < cutoff_out_max + cutoff_out_step:
            if cutoff_out not in evaluation.keys():
                evaluation[cutoff_out] = {}

            select = outliers_pixels(img, cutoff_out, n)
            score = compute_score_lavan(img, select, n)
            # score = compute_score(img, mode=mode)
            if is_normalized:
                score = normalize(score)

            cutoff = cutoff_min
            while cutoff < cutoff_max + cutoff_step:
                if cutoff not in evaluation[cutoff_out].keys():
                    evaluation[cutoff_out][cutoff] = {'TP': 0, 'FP': 0, 'FN': 0}

                for i in range(score.shape[0]):
                    for j in range(score.shape[1]):

                        if (score[i][j] >= cutoff
                                and i in range(patch_pointer[0], patch_pointer[0] + patch_side)
                                and j in range(patch_pointer[1], patch_pointer[1] + patch_side)):
                            evaluation[cutoff_out][cutoff]['TP'] += 1

                        elif (score[i][j] >= cutoff
                              and ((i not in range(patch_pointer[0], patch_pointer[0] + patch_side))
                                   or (j not in range(patch_pointer[1], patch_pointer[1] + patch_side)))):
                            evaluation[cutoff_out][cutoff]['FP'] += 1

                        elif (score[i][j] < cutoff
                              and i in range(patch_pointer[0], patch_pointer[0] + patch_side)
                              and j in range(patch_pointer[1], patch_pointer[1] + patch_side)):
                            evaluation[cutoff_out][cutoff]['FN'] += 1

                cutoff += cutoff_step
            cutoff_out += cutoff_out_step
        num += 1
        print(f"Processing: {round(num * 100 / len(files), 8)}%")

    for cutoff_out in evaluation.keys():
        for cutoff in evaluation[cutoff_out].keys():
            evaluation[cutoff_out][cutoff]['P'] = compute_precision(TP=evaluation[cutoff_out][cutoff]['TP'],
                                                                    FP=evaluation[cutoff_out][cutoff]['FP'])
            evaluation[cutoff_out][cutoff]['R'] = compute_recall(TP=evaluation[cutoff_out][cutoff]['TP'],
                                                                 FN=evaluation[cutoff_out][cutoff]['FN'])
            evaluation[cutoff_out][cutoff]['F'] = compute_F(precision=evaluation[cutoff_out][cutoff]['P'],
                                                            recall=evaluation[cutoff_out][cutoff]['R'])

    if save:
        text = "cutoff_out, cutoff, TP, FP, FN, Precision, Recall, F1-score\n"
        for cutoff_out in evaluation.keys():
            for cutoff in evaluation[cutoff_out].keys():
                text += f"{cutoff_out},{cutoff},{evaluation[cutoff_out][cutoff]['TP']},{evaluation[cutoff_out][cutoff]['FP']},{evaluation[cutoff_out][cutoff]['FN']},{evaluation[cutoff_out][cutoff]['P']},{evaluation[cutoff_out][cutoff]['R']},{evaluation[cutoff_out][cutoff]['F']}\n"

        stat = open("patch_detection_stat_new.txt", 'w')
        stat.write(text)
        stat.close()

    print("test_patch_detection is finished")
    return evaluation


def test_lavan_attack_detection(dir, cutoff_out_min=0, cutoff_out_max=10, cutoff_out_step=0.1,
                                cutoff_min=0, cutoff_max=10, cutoff_step=0.1, save=True):
    files = [f for f in listdir(dir)]
    files = files[:1000]
    num = 0
    evaluation = {}
    for f in files:
        print(f)

        img = cv2.imread(join(dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cutoff_out = cutoff_out_min

        while cutoff_out < cutoff_out_max + cutoff_out_step:
            if cutoff_out not in evaluation.keys():
                evaluation[cutoff_out] = {}

            near_n = compute_nearest_neighbor(img)

            pixels = outliers_pixels(img, cutoff_out, near_n)

            score = compute_score_lavan(img, pixels, near_n)

            cutoff = cutoff_min
            while cutoff < cutoff_max + cutoff_step:
                if cutoff not in evaluation[cutoff_out].keys():
                    evaluation[cutoff_out][cutoff] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

                if 'adversarial' in f:
                    if check_attack(score, cutoff):
                        evaluation[cutoff_out][cutoff]['TP'] += 1
                    else:
                        evaluation[cutoff_out][cutoff]['FN'] += 1

                elif 'original' in f:
                    if check_attack(score, cutoff):
                        evaluation[cutoff_out][cutoff]['FP'] += 1
                    else:
                        evaluation[cutoff_out][cutoff]['TN'] += 1

                else:
                    print(f"Unexpected filename: {f} - no adversity marker")

                cutoff += cutoff_step
            cutoff_out += cutoff_out_step

        num += 1
        print(f"Processing: {round(num * 100 / len(files), 8)}%")

    for cutoff_out in evaluation.keys():
        for cutoff in evaluation[cutoff_out].keys():
            evaluation[cutoff_out][cutoff]['ACC'] = compute_accuracy(TP=evaluation[cutoff_out][cutoff]['TP'],
                                                                     FP=evaluation[cutoff_out][cutoff]['FP'],
                                                                     FN=evaluation[cutoff_out][cutoff]['FN'],
                                                                     TN=evaluation[cutoff_out][cutoff]['TN'])
            evaluation[cutoff_out][cutoff]['P'] = compute_precision(TP=evaluation[cutoff_out][cutoff]['TP'],
                                                                    FP=evaluation[cutoff_out][cutoff]['FP'])
            evaluation[cutoff_out][cutoff]['R'] = compute_recall(TP=evaluation[cutoff_out][cutoff]['TP'],
                                                                 FN=evaluation[cutoff_out][cutoff]['FN'])
            evaluation[cutoff_out][cutoff]['F'] = compute_F(precision=evaluation[cutoff_out][cutoff]['P'],
                                                            recall=evaluation[cutoff_out][cutoff]['R'])

    if save:
        text = "cutoff_out, cutoff, TP, FP, FN, TN, Accuracy, Precision, Recall, F1-score\n"
        for cutoff_out in evaluation.keys():
            for cutoff in evaluation[cutoff_out].keys():
                text += f"{cutoff_out},{cutoff},{evaluation[cutoff_out][cutoff]['TP']},{evaluation[cutoff_out][cutoff]['FP']},{evaluation[cutoff_out][cutoff]['FN']},{evaluation[cutoff_out][cutoff]['TN']},{evaluation[cutoff_out][cutoff]['ACC']},{evaluation[cutoff_out][cutoff]['P']},{evaluation[cutoff_out][cutoff]['R']},{evaluation[cutoff_out][cutoff]['F']}\n"

        stat = open("attack_lavan_detection_stat_10.txt", 'w')
        stat.write(text)
        stat.close()

    print("test_attack_detection is finished")
    return evaluation


def test_lavan_patch_detection(dir, patch_pointer=(0, 0), patch_side=10,
                               cutoff_border_min=0, cutoff_border_max=10, cutoff_border_step=0.1,
                               cutoff_out_min=0, cutoff_out_max=3, cutoff_out_step=0.1,
                               is_normalized=False, save=True):
    # patch_pointer - top left angle
    # patch_side - width and height of the patch

    files = [f for f in listdir(dir) if 'adversarial' in f]
    files = files[:50]
    num = 0
    evaluation = {}
    for f in files:
        img = cv2.imread(join(dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        near_n = compute_nearest_neighbor(img)

        cutoff_out = cutoff_out_min
        while cutoff_out < cutoff_out_max + cutoff_out_step:
            pixels = outliers_pixels(img, cutoff_out, near_n)
            score = compute_score_lavan(img, pixels, near_n)

            if cutoff_out not in evaluation.keys():
                evaluation[cutoff_out] = {}

            cutoff_border = cutoff_border_min
            while cutoff_border < (cutoff_border_max + cutoff_border_step):
                if cutoff_border not in evaluation[cutoff_out].keys():
                    evaluation[cutoff_out][cutoff_border] = {'TP': 0, 'FP': 0, 'FN': 0}

                borders = find_borders(score, cutoff_border)

                area = check_patch(borders, img)

                for i in range(score.shape[0]):
                    for j in range(score.shape[1]):

                        if (area[i][j] == 1
                                and i in range(patch_pointer[0], patch_pointer[0] + patch_side)
                                and j in range(patch_pointer[1], patch_pointer[1] + patch_side)):
                            evaluation[cutoff_out][cutoff_border]['TP'] += 1

                        elif (area[i][j] == 1
                              and (i not in range(patch_pointer[0], patch_pointer[0] + patch_side)
                                   or j not in range(patch_pointer[1], patch_pointer[1] + patch_side))):
                            evaluation[cutoff_out][cutoff_border]['FP'] += 1

                        elif (area[i][j] == 0
                              and i in range(patch_pointer[0], patch_pointer[0] + patch_side)
                              and j in range(patch_pointer[1], patch_pointer[1] + patch_side)):
                            evaluation[cutoff_out][cutoff_border]['FN'] += 1

                cutoff_border += cutoff_border_step
            cutoff_out += cutoff_out_step

        num += 1
        print(f"Processing: {round(num * 100 / len(files), 2)}%")

    for cutoff_out in evaluation.keys():
        for cutoff_border in evaluation[cutoff_out].keys():
            evaluation[cutoff_out][cutoff_border]['P'] = compute_precision(
                TP=evaluation[cutoff_out][cutoff_border]['TP'],
                FP=evaluation[cutoff_out][cutoff_border]['FP'])
            evaluation[cutoff_out][cutoff_border]['R'] = compute_recall(TP=evaluation[cutoff_out][cutoff_border]['TP'],
                                                                        FN=evaluation[cutoff_out][cutoff_border]['FN'])
            evaluation[cutoff_out][cutoff_border]['F'] = compute_F(precision=evaluation[cutoff_out][cutoff_border]['P'],
                                                                   recall=evaluation[cutoff_out][cutoff_border]['R'])

    if save:
        text = "cutoff_out, cutoff_border, TP, FP, FN, Precision, Recall, F1-score\n"
        for cutoff_out in evaluation.keys():
            for cutoff_border in evaluation[cutoff_out].keys():
                text += f"{cutoff_out},{cutoff_border},{evaluation[cutoff_out][cutoff_border]['TP']},{evaluation[cutoff_out][cutoff_border]['FP']},{evaluation[cutoff_out][cutoff_border]['FN']},{evaluation[cutoff_out][cutoff_border]['P']},{evaluation[cutoff_out][cutoff_border]['R']},{evaluation[cutoff_out][cutoff_border]['F']}\n"

        stat = open("patch_lavan_detection_stat.txt", 'w')
        stat.write(text)
        stat.close()

    print("test_patch_detection is finished")
    return evaluation


def test_lavan_is_patched_detection(dir_patch, dir_1px, dir_jsma,
                                    cutoff_min=0, cutoff_max=10, cutoff_step=0.1,
                                    cutoff_dist_min=0, cutoff_dist_max=1, cutoff_dist_step=0.1,
                                    is_normalized=False, save=True):
    # patch_pointer - top left angle
    # patch_side - width and height of the patch

    files_patch = [f for f in listdir(dir_patch) if 'adversarial' in f]
    files_1px = [f for f in listdir(dir_1px)]
    files_jsma = [f for f in listdir(dir_jsma)]
    files = files_patch[:10000] + files_1px[:10000] + files_jsma[:10000]
    num = 0
    evaluation = {}
    for f in files:
        if 'adversarial' in f:
            img = cv2.imread(join(dir_patch, f))
        elif 'jsma' in f:
            img = cv2.imread(join(dir_jsma, f))
        else:
            img = cv2.imread(join(dir_1px, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score = compute_score(img, 'mn')

        cutoff_dist = cutoff_dist_min
        while cutoff_dist < cutoff_dist_max + cutoff_dist_step:
            if cutoff_dist not in evaluation.keys():
                evaluation[cutoff_dist] = {}

            cutoff = cutoff_min
            while cutoff < cutoff_max + cutoff_step:
                if cutoff not in evaluation[cutoff_dist].keys():
                    evaluation[cutoff_dist][cutoff] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

                result = is_patched(img, score, cutoff, cutoff_dist)

                if result and 'adversarial' in f:
                    evaluation[cutoff_dist][cutoff]['TP'] += 1
                elif result:
                    evaluation[cutoff_dist][cutoff]['FP'] += 1
                elif (not result) and 'adversarial' in f:
                    evaluation[cutoff_dist][cutoff]['FN'] += 1
                elif not result:
                    evaluation[cutoff_dist][cutoff]['TN'] += 1
                else:
                    print(f"Unexpected filename: {f} - no adversity marker")

                cutoff += cutoff_step
            cutoff_dist += cutoff_dist_step
        num += 1
        print(f"Processing: {round(num * 100 / len(files), 8)}%")

        for cutoff_dist in evaluation.keys():
            for cutoff in evaluation[cutoff_dist].keys():
                evaluation[cutoff_dist][cutoff]['ACC'] = compute_accuracy(TP=evaluation[cutoff_dist][cutoff]['TP'],
                                                                          FP=evaluation[cutoff_dist][cutoff]['FP'],
                                                                          FN=evaluation[cutoff_dist][cutoff]['FN'],
                                                                          TN=evaluation[cutoff_dist][cutoff]['TN'])
                evaluation[cutoff_dist][cutoff]['P'] = compute_precision(TP=evaluation[cutoff_dist][cutoff]['TP'],
                                                                         FP=evaluation[cutoff_dist][cutoff]['FP'])
                evaluation[cutoff_dist][cutoff]['R'] = compute_recall(TP=evaluation[cutoff_dist][cutoff]['TP'],
                                                                      FN=evaluation[cutoff_dist][cutoff]['FN'])
                evaluation[cutoff_dist][cutoff]['F'] = compute_F(precision=evaluation[cutoff_dist][cutoff]['P'],
                                                                 recall=evaluation[cutoff_dist][cutoff]['R'])

        if save:
            text = "cutoff_dist, cutoff, TP, FP, FN, TN, Accuracy, Precision, Recall, F1-score\n"
            for cutoff_dist in evaluation.keys():
                for cutoff in evaluation[cutoff_dist].keys():
                    text += f"{cutoff_dist},{cutoff},{evaluation[cutoff_dist][cutoff]['TP']},{evaluation[cutoff_dist][cutoff]['FP']},{evaluation[cutoff_dist][cutoff]['FN']},{evaluation[cutoff_dist][cutoff]['TN']},{evaluation[cutoff_dist][cutoff]['ACC']},{evaluation[cutoff_dist][cutoff]['P']},{evaluation[cutoff_dist][cutoff]['R']},{evaluation[cutoff_dist][cutoff]['F']}\n"

            stat = open("is_patched_detection_stat.txt", 'w')
            stat.write(text)
            stat.close()

    print("test_patch_detection is finished")
    return evaluation


dir = "D:/PyCharm Projects/LaVan/logs"
dir_1px = "D:/PyCharm Projects/1px_detection/dataset/c_4test/1px/attacked images"
dir_jsma = "D:/PyCharm Projects/1px_detection/dataset/c_4test/jsma 005/attacked images"
test_attack_detection(dir=dir, cutoff_min=0, cutoff_max=15, cutoff_step=0.1, mode='mn', save=True)
# pixels = test_patch_detection(dir=dir, patch_pointer=(210, 210), patch_side=50,
#                               cutoff_min=0, cutoff_max=10, cutoff_step=0.1,
#                               mode='mn', is_normalized=False, save=True)
# pixels_new = test_patch_detection_new(dir=dir, patch_pointer=(210, 210), patch_side=50,
#                                       cutoff_min=0, cutoff_max=10, cutoff_step=0.1,
#                                       cutoff_out_min=0.1, cutoff_out_max=3, cutoff_out_step=0.1,
#                                       mode='mn', is_normalized=False, save=True)
# test_lavan_attack_detection(dir=dir, cutoff_out_min=0.1, cutoff_out_max=0.1, cutoff_out_step=0.1, cutoff_min=8, cutoff_max=16,
#                             cutoff_step=0.1, save=True)
'''test_lavan_patch_detection(dir=dir, patch_pointer=(210, 210), patch_side=50,
                           cutoff_border_min=50, cutoff_border_max=150, cutoff_border_step=1,
                           cutoff_out_min=0, cutoff_out_max=3, cutoff_out_step=1,
                           is_normalized=False, save=True)'''

# t_start = time.time()
# test_lavan_is_patched_detection(dir_patch=dir, dir_1px=dir_1px, dir_jsma=dir_jsma,
#                                 cutoff_min=4, cutoff_max=6, cutoff_step=0.01,
#                                 cutoff_dist_min=0.07, cutoff_dist_max=0.11, cutoff_dist_step=0.01,
#                                 is_normalized=False, save=True)
# t_stop = time.time()
# print(t_stop - t_start)

print("Done")
