import os.path

import numpy as np
import json
import time

JNUM = 21
skeleton_path = '../../data/assemblyhands/annotations/skeleton.txt'


def load_skeleton(path, joint_num):
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id

    return skeleton


def num_check(path):
    # skeleton = load_skeleton(skeleton_path, JNUM * 2)
    lines = open(path).readlines()
    lines.pop(0)
    lines.pop(-1)
    print(len(lines))

    mpjpe = [[] for _ in range(JNUM)]
    left_num, right_num = 0, 0

    error = []

    for line in lines:
        line = line.split()

        # print(len(line))
        seq, cam, frame = line[1], line[2], int(line[3])
        pred = np.array(line[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        gt = np.array(line[5].split(',')).astype(np.float64).reshape(JNUM, 3)
        valid = np.array(line[6].split(',')).astype(np.float64).reshape(JNUM, )
        rot = np.array(line[7].split(',')).astype(np.float64).reshape(3, 3)
        pos = np.array(line[8].split(',')).astype(np.float64).reshape(3, )
        hand = line[11]
        # handpred = np.array(line[12].split(',')).astype(np.float64).reshape(2, )

        # print(valid)
        # print(gt[:3])
        if hand == 'left':
            left_num += 1
        if hand == 'right':
            right_num += 1

        # mpjpe
        for j in range(JNUM):
            if valid[j]:
                # mpjpe[j].append(np.sqrt(np.sum((pred[j] - gt[j]) ** 2)))
                mpjpe[j].append(np.linalg.norm(pred[j] - gt[j]))
        error.append(mpjpe_per_hand(pred, gt, valid))

    sum, num = 0, 0
    for j in range(JNUM):
        length = len(mpjpe[j])
        if length > 0:
            num += len(mpjpe[j])
            sum += np.sum(mpjpe[j])
            mpjpe[j] = np.mean(mpjpe[j])
        else:
            mpjpe[j] = np.nan
        # joint_name = skeleton[j]['name']
        print(f'{length},{mpjpe[j]:.2f} ', end='\n')
    # print(f'\nMPJPE for single hand sequences: {np.nanmean(error):.2f}, left:{left_num}, right:{right_num}')
    print(f'\nMPJPE for single hand sequences: {np.nanmean(error):.2f} s/n{sum/num:.2f} nanmean{np.nanmean(mpjpe):.2f},'
          f' left:{left_num}, right:{right_num}')


def gene_json_from_log(log_path):
    tic = time.time()
    lines = open(log_path).readlines()
    lines.pop(0)
    lines.pop(-1)
    print(len(lines))

    test_result = {}

    for line in lines[:]:
        line = line.split()

        # print(len(line))
        img, seq, cam, frame = line[0], line[1], line[2], int(line[3])
        pred = np.array(line[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        gt = np.array(line[5].split(',')).astype(np.float64).reshape(JNUM, 3)
        valid = np.array(line[6].split(',')).astype(np.float64).reshape(JNUM, )
        rot = np.array(line[7].split(',')).astype(np.float64).reshape(3, 3)
        pos = np.array(line[8].split(',')).astype(np.float64).reshape(3, )
        focal = np.array(line[9].split(',')).astype(np.float64).reshape(2, )
        pricpt = np.array(line[10].split(',')).astype(np.float64).reshape(2, )
        hand = line[11]
        # handpred = np.array(line[12].split(',')).astype(np.float64).reshape(2, )
        # abs_depth = np.array(line[13].split(',')).astype(np.float64).reshape(2, )
        # depth_info = np.array(line[14].split(',')).astype(np.float64).reshape(2, 3)

        action_dic = test_result.get(seq, {})
        frame_dic = action_dic.get(frame, {})

        gt_ls = frame_dic.get(hand, [])
        # gt_ls.append([img, cam, gt.tolist(), pred.tolist(), rot.tolist(), valid.tolist(), focal.tolist(),
        #               pricpt.tolist()])
        gt_ls.append([cam, gt.tolist(), pred.tolist(), rot.tolist(), valid.tolist()])
        frame_dic[hand] = gt_ls

        action_dic[frame] = frame_dic
        test_result[seq] = action_dic

    with open(f'{log_path}.json', 'w') as fw:
        json.dump(test_result, fw, indent=4)

    print(f'time consumed:{time.time() - tic:.2f}')


def mpjpe_per_hand(pred, gt, valid):
    assert pred.shape[0] == gt.shape[0] == valid.shape[0]
    assert pred.shape in [(JNUM, 3), (3, JNUM)]
    if pred.shape == (3, JNUM):
        pred, gt = pred.T, gt.T

    mpjpe = []
    for j in range(JNUM):
        pred_j, gt_j = pred[j], gt[j]
        mpjpe_j = np.linalg.norm(pred_j - gt_j)
        if valid[j]:
            mpjpe.append(mpjpe_j)
        else:
            mpjpe.append(np.nan)

    return mpjpe


def avg_pred(preds, valids):
    assert len(preds) == len(valids)
    preds_nan, valids_nan = np.array(preds), np.array(valids)

    valids_nan[valids_nan < 0.5] = np.nan
    preds_nan *= valids_nan

    return np.nanmean(preds_nan, axis=0)


def merge_result(path):
    '''
    Validate whether using multi-view predictions can benefit the accuracy.
    1. Transform all the predictions to the same world coordinate system.
    2. Calculate the final prediction by average.
    3. Calculate the MPJPE using the final prediction.
    '''
    tic = time.time()
    test_result = json.load(open(path))
    # print(f'time consumed:{time.time() - tic:.2f}')

    mpjpe_ls = []
    for action in test_result:
        for frame in test_result[action]:
            # print(test_result[action][frame])
            for hand in test_result[action][frame]:
                data_ls = test_result[action][frame][hand]

                # Non multi-view sample
                if len(data_ls) == 1:
                    cam, gt, pred, R, valid = data_ls[0]
                    gt, pred, R, valid = np.array(gt), np.array(pred), np.array(R), np.array(valid).reshape(JNUM, 1)
                    mpjpe_ls.append(mpjpe_per_hand(pred, gt, valid))
                    continue

                # Get predictions, gts, and valid under multiple views
                preds_world = []
                valids = []
                gt_world = None
                for cam, gt, pred, R, valid in data_ls:
                    gt, pred, R, valid = np.array(gt), np.array(pred), np.array(R), np.array(valid).reshape(JNUM, 1)
                    preds_world.append(pred @ R)  # (R.T @ pred.T).T
                    valids.append(valid)
                    if gt_world is None:
                        gt_world = gt @ R  # (R.T @ gt.T).T
                # Calculate the final prediction by average
                avg_pred_world = avg_pred(preds_world, valids)
                valid_world = np.sum(valids, axis=0)

                # Keep the total number
                for _ in range(len(preds_world)):
                    mpjpe_ls.append(mpjpe_per_hand(avg_pred_world, gt_world, valid_world))

    print(f'MPJPE after merging the outputs under multiple views:{np.nanmean(mpjpe_ls):.2f}')
    return np.nanmean(mpjpe_ls), len(mpjpe_ls)


def R_from_2poses(joints0, joints1):
    '''
    input: joints0 & joints1
    output: rotation matrix R, CS0->CS1
    '''
    joints0, joints1 = np.array(joints0), np.array(joints1)
    Cov = joints0.T @ joints1
    U, S, Vt = np.linalg.svd(Cov)
    return U @ Vt


def merge_result_wo_gt(path):
    '''
    Validate whether using multi-view predictions can benefit the accuracy.
    1. Transform all the predictions to the same world coordinate system.
    2. Calculate the final prediction by average.
    3. Calculate the MPJPE using the final prediction.
    '''
    tic = time.time()
    test_result = json.load(open(path))
    # print(f'time consumed:{time.time() - tic:.2f}')

    mpjpe_ls = []
    for action in test_result:
        for frame in test_result[action]:
            # print(test_result[action][frame])
            for hand in test_result[action][frame]:
                data_ls = test_result[action][frame][hand]

                # Non multi-view sample
                if len(data_ls) == 1:
                    cam, gt, pred, R, valid = data_ls[0]
                    gt, pred, R, valid = np.array(gt), np.array(pred), np.array(R), np.array(valid).reshape(JNUM, 1)
                    mpjpe_ls.append(mpjpe_per_hand(pred, gt, valid))
                    continue

                # Get predictions, gts, and valid under multiple views
                cam0, gt0, pred0, R0, valid0 = data_ls[0][0:5]
                cam1, gt1, pred1, R1, valid1 = data_ls[1][0:5]
                gt0, pred0, R0, valid0 = np.array(gt0), np.array(pred0), np.array(R0), np.array(valid0).reshape(
                    JNUM, 1)
                gt1, pred1, R1, valid1 = np.array(gt1), np.array(pred1), np.array(R1), np.array(valid1).reshape(
                    JNUM, 1)

                p0, p1 = [], []
                for i in range(pred0.shape[0]):
                    if valid0[i] and valid1[i]:
                        p0.append(pred0[i])
                        p1.append(pred1[i])
                R01 = R_from_2poses(p0, p1)
                R10 = R_from_2poses(p1, p0)

                preds_cs0, preds_cs1 = [], []
                valids_cs0, valids_cs1 = [], []

                preds_cs0.append(pred0)
                preds_cs0.append(pred1 @ R10)
                preds_cs1.append(pred0 @ R01)
                preds_cs1.append(pred1)

                valids_cs0.append(valid0)
                valids_cs0.append(valid1)
                valids_cs1.append(valid0)
                valids_cs1.append(valid1)

                avg_pred_cs0 = avg_pred(preds_cs0, valids_cs0)
                avg_pred_cs1 = avg_pred(preds_cs1, valids_cs1)

                valid_cs0 = np.sum(valids_cs0, axis=0)
                valid_cs1 = np.sum(valids_cs1, axis=0)

                mpjpe_ls.append(mpjpe_per_hand(avg_pred_cs0, gt0, valid_cs0))
                mpjpe_ls.append(mpjpe_per_hand(avg_pred_cs1, gt1, valid_cs1))

    print(f'MPJPE after merging the outputs under multiple views:{np.nanmean(mpjpe_ls):.2f}')
    return np.nanmean(mpjpe_ls), len(mpjpe_ls)


def count_pair_num(path):
    tic = time.time()
    test_result = json.load(open(path))
    print(f'time consumed:{time.time() - tic:.2f}')

    cam_name = set()
    cam_combines = []
    for action in test_result:
        for frame in test_result[action]:
            # print(test_result[action][frame])
            for hand in test_result[action][frame]:
                data_ls = test_result[action][frame][hand]
                cams = [data[0] for data in data_ls]
                for cam in cams:
                    cam_name.add(cam)
                cams = sorted(cams, key=lambda x: x)
                cam_combines.append(' '.join(cams))

    count = {}
    for cams in cam_combines:
        num = count.get(cams, 0)
        num += 1
        count[cams] = num
    print(count)
    print(cam_name)


def eval_pair(log, setup=0, pair=(0, 1)):
    jsonpath = f'{log}.json'
    if not os.path.exists(jsonpath):
        print('generate json...', end='')
        tic = time.time()
        gene_json_from_log(log)
        print(f'done. Time consumed:{time.time() - tic:.2f}s')

    setups = [['HMC_21176875', 'HMC_21176623', 'HMC_21110305', 'HMC_21179183'],
              ['HMC_84346135', 'HMC_84347414', 'HMC_84355350', 'HMC_84358933']]
    cam_pair = [setups[setup][pair[0]], setups[setup][pair[1]]]

    print('load json...', end='')
    tic = time.time()
    test_result = json.load(open(f'{log}.json'))
    print(f'done. Time consumed:{time.time() - tic:.2f}s')

    mpjpe_ls = []
    for action in test_result:
        for frame in test_result[action]:
            # print(test_result[action][frame])
            for hand in test_result[action][frame]:
                data_ls = test_result[action][frame][hand]

                # Non multi-view sample
                if len(data_ls) == 1:
                    continue

                cam_ls = [x[0] for x in data_ls]
                try:
                    idx = [cam_ls.index(cam) for cam in cam_pair]
                    # print(idx)
                except:
                    continue
                # Get predictions, gts, and valid under multiple views
                cam1, gt1, pred1, R1, valid1 = data_ls[idx[0]]
                cam2, gt2, pred2, R2, valid2 = data_ls[idx[1]]
                gt1, pred1, R1, valid1 = np.array(gt1), np.array(pred1), np.array(R1), np.array(valid1).reshape(
                    JNUM, 1)
                gt2, pred2, R2, valid2 = np.array(gt2), np.array(pred2), np.array(R2), np.array(valid2).reshape(
                    JNUM, 1)

                mpjpe_ls.append(mpjpe_per_hand(pred1, gt1, valid1))
                mpjpe_ls.append(mpjpe_per_hand(pred2, gt2, valid2))

    print(f'MPJPE for setup{setup}, pair{pair}:{np.nanmean(mpjpe_ls):.2f}, num:{len(mpjpe_ls)}')
    return np.nanmean(mpjpe_ls), len(mpjpe_ls)


if __name__ == '__main__':
    import glob
    ls = glob.glob('./checkpoints0/evaluation/ah/28.log.json')
    ls.sort()
    num_ls, mp_ls = [], []
    for j in ls:
        print(j)
        merge_result_wo_gt(j)
