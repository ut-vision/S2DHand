import glob
import json
import os.path

import numpy as np
import time
from utils.func import R_from_2poses, merge_from_2hands, matrix_from_quaternion, quaternion_from_matrix
from consistency import mpjpe_per_hand

JNUM = 21


def R_from_2hands(hands1, hands2, valids1, valids2, R12=None, R21=None):
    for i in range(hands1.shape[0]):
        pred1, pred2 = hands1[i], hands2[i]
        p1, p2 = [], []
        for j in range(pred1.shape[0]):
            if valids1[i, j] and valids2[i, j]:
                p1.append(pred1[j])
                p2.append(pred2[j])
        if R12 is None:
            R12 = R_from_2poses(p1, p2, is_torch=False)
        if R21 is None:
            R21 = R_from_2poses(p2, p1, is_torch=False)

    return R12, R21


def matrix2angle(m):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY, the object will be rotated with the order of [rx, ry, rz]
    """
    x = np.arctan2(m[2, 1], m[2, 2])
    y = np.arcsin(-m[2, 0])
    z = np.arctan2(m[1, 0], m[0, 0])
    return np.array([x, y, z])


def calc_R(log_path):
    lines = open(log_path).readlines()
    lines.pop(0)
    mono = float(lines.pop(-1).split(':')[-1])
    line_num = len(lines)

    R_ls, R_gt = [], None

    for i in range(min(512, line_num // 2)):
        view0 = lines[2 * i].split()
        view1 = lines[2 * i + 1].split()

        pred0 = np.array(view0[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        valid0 = np.array(view0[6].split(',')).astype(np.float64).reshape(JNUM, )

        pred1 = np.array(view1[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        valid1 = np.array(view1[6].split(',')).astype(np.float64).reshape(JNUM, )

        R_pred, _ = R_from_2hands(pred0[np.newaxis], pred1[np.newaxis],
                                  valid0[np.newaxis, :], valid1[np.newaxis, :])
        R_ls.append(R_pred)

    quan_ls = [quaternion_from_matrix(R) for R in R_ls]
    quan_avg = np.mean(quan_ls, axis=0)
    quan_avg /= np.linalg.norm(quan_avg)

    R_est = matrix_from_quaternion(np.mean(quan_ls, axis=0))
    return R_est


def calc_merged_metric(log_path, use_gt=False, R12=None, R21=None, dynamic_R=False):
    if dynamic_R:
        print('calculating new R...')
        R12 = calc_R(log_path)
        R21 = R12.T
        print('done')
    name = os.path.basename(log_path).split('.')[0]

    lines = open(log_path).readlines()
    lines.pop(0)
    mono = float(lines.pop(-1).split(':')[-1])
    line_num = len(lines)
    sumf = []

    mpjpe_ls, q_diff_ls = [], []
    for i in range(line_num // 2):
        view0 = lines[2 * i].split()
        view1 = lines[2 * i + 1].split()

        pred0 = np.array(view0[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        gt0 = np.array(view0[5].split(',')).astype(np.float64).reshape(JNUM, 3)
        R0 = np.array(view0[7].split(',')).astype(np.float64).reshape(3, 3)
        valid0 = np.array(view0[6].split(',')).astype(np.float64).reshape(JNUM, )

        pred1 = np.array(view1[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        gt1 = np.array(view1[5].split(',')).astype(np.float64).reshape(JNUM, 3)
        R1 = np.array(view1[7].split(',')).astype(np.float64).reshape(3, 3)
        valid1 = np.array(view1[6].split(',')).astype(np.float64).reshape(JNUM, )

        R = R_from_2poses(pred0, pred1)
        R_gt = R0 @ R1.T
        q_diff = matrix2angle(R) - matrix2angle(R_gt)
        q_diff_ls.append(q_diff)

        if not use_gt:
            merge0, merge1 = merge_from_2hands(pred0[np.newaxis, :, :], pred1[np.newaxis, :, :],
                                                valid0[np.newaxis, :], valid1[np.newaxis, :])
        else:
            if R12 is None:
                R12 = R0 @ R1.T
            if R21 is None:
                R21 = R1 @ R0.T
            merge0, merge1 = merge_from_2hands(pred0[np.newaxis, :, :], pred1[np.newaxis, :, :],
                                                valid0[np.newaxis, :], valid1[np.newaxis, :],
                                                R12=R12, R21=R21)
        merge0, merge1 = np.squeeze(merge0), np.squeeze(merge1)

        # f0 = np.array(mpjpe_per_hand(merge0, gt0, valid0)) < np.array(mpjpe_per_hand(pred0, gt0, valid0))
        # f1 = np.array(mpjpe_per_hand(merge1, gt1, valid1)) < np.array(mpjpe_per_hand(pred1, gt1, valid1))
        # sumf.append(sum(f0 == f1))

        mpjpe_ls.append(mpjpe_per_hand(merge0, gt0, valid0))
        mpjpe_ls.append(mpjpe_per_hand(merge1, gt1, valid1))

    q_mean = np.mean(np.abs(q_diff_ls), axis=0)
    q_mean_str = ','.join([f'{u:.2f}' for u in q_mean])
    print(f'{name} num:{len(mpjpe_ls)}, mono:{mono:.2f}, merge:{np.nanmean(mpjpe_ls):.2f}, '
          f'q_mean:({q_mean_str})')
    # print(np.mean(sumf))
    return len(mpjpe_ls), mono, np.nanmean(mpjpe_ls), q_diff_ls


if __name__ == '__main__':
    logs = glob.glob(
        '/large/lruicong/cvpr24/DualMini/ablation/no_momentum/*/evaluation/ah/*-set1-0,1.log')
    logs.sort()
    print(logs)
    R_config = json.load(open('/large/lruicong/cvpr24/DualMini/R_config.json'))

    q_diffs = []
    num, mono, merge = 0, 0, 0
    for log in logs:
        print(log)
        setup, pair = os.path.basename(log).split('.')[0].split('-')[1:]
        R12 = np.array(R_config[f'{setup}-{pair}']['R_pred'])

        n, mo, mer, q_diff = calc_merged_metric(log, use_gt=True, R12=R12, R21=R12.T, dynamic_R=False)
        num += n
        mono += mo * n
        merge += mer * n
        q_diffs += q_diff
    print(f'total:{num}, mean mono:{mono / num:.2f}, mean merge:{merge / num:.2f}')

    # q_diffs = np.array(q_diffs)
    # print(q_diffs.shape)
    # np.save('/large/lruicong/temp/r.npy', q_diffs)
