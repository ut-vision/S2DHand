import numpy as np


def align_two_pred(pred1, pred2, root_idx=0):
    pred1_ = pred1.copy()
    pred2_ = pred2.copy()

    # gtj :B*21*3
    # prj :B*21*3
    # root_idx = 0  # root
    ref_bone_link = [0, 9]  # mid mcp
    pred2_align = pred2_.copy()

    pred1_anchor = pred1[:, root_idx: root_idx + 1, :].copy()
    pred2_anchor = pred2[:, root_idx: root_idx + 1, :].copy()

    scales2 = []
    for i in range(pred2_.shape[0]):

        pred_ref_bone_len = np.linalg.norm(pred2_[i][ref_bone_link[0]] - pred2_[i][ref_bone_link[1]])
        gt_ref_bone_len = np.linalg.norm(pred1_[i][ref_bone_link[0]] - pred1_[i][ref_bone_link[1]])
        scale = gt_ref_bone_len / pred_ref_bone_len
        scales2.append(scale)

        for j in range(21):
            pred2_align[i][j] = pred1_[i][root_idx] + scale * (pred2_[i][j] - pred2_[i][root_idx])

    pred1_ -= pred1[:, root_idx: root_idx + 1, :]
    pred2_align -= pred2_align[:, root_idx: root_idx + 1, :]

    return pred1_, pred2_align, pred1_anchor, pred2_anchor, np.array(scales2)[:, np.newaxis, np.newaxis]


def global_align(gtj0, prj0, key, root_idx=0):
    gtj = gtj0.copy()
    prj = prj0.copy()

    if key in ["stb", "rhd", "ah"]:
        # gtj :B*21*3
        # prj :B*21*3
        # root_idx = 0  # root
        ref_bone_link = [0, 9]  # mid mcp
        pred_align = prj.copy()
        for i in range(prj.shape[0]):

            pred_ref_bone_len = np.linalg.norm(prj[i][ref_bone_link[0]] - prj[i][ref_bone_link[1]])
            gt_ref_bone_len = np.linalg.norm(gtj[i][ref_bone_link[0]] - gtj[i][ref_bone_link[1]])
            scale = gt_ref_bone_len / pred_ref_bone_len
            # scale = 0.0824 / pred_ref_bone_len

            for j in range(21):
                pred_align[i][j] = gtj[i][root_idx] + scale * (prj[i][j] - prj[i][root_idx])

        gtj_wrist_align = gtj - gtj0[:, root_idx: root_idx + 1, :]
        pred_wrist_align = pred_align - pred_align[:, root_idx: root_idx + 1, :]

        return gtj_wrist_align, pred_wrist_align, gtj, pred_align

    if key in ["do", "eo"]:
        # gtj :B*5*3
        # prj :B*5*3

        prj_ = prj.copy()[:, [4, 8, 12, 16, 20], :]  # B*5*3

        gtj_valid = []
        prj_valid_align = []

        for i in range(prj_.shape[0]):
            # 5*3
            mask = ~(np.isnan(gtj[i][:, 0]))
            if mask.sum() < 2:
                continue

            prj_mask = prj_[i][mask]  # m*3
            gtj_mask = gtj[i][mask]  # m*3

            gtj_valid_center = np.mean(gtj_mask, 0)
            prj_valid_center = np.mean(prj_mask, 0)

            gtj_center_length = np.linalg.norm(gtj_mask - gtj_valid_center, axis=1).mean()
            prj_center_length = np.linalg.norm(prj_mask - prj_valid_center, axis=1).mean()
            scale = gtj_center_length / prj_center_length

            prj_valid_align_i = gtj_valid_center + scale * (prj_[i][mask] - prj_valid_center)

            gtj_valid.append(gtj_mask)
            prj_valid_align.append(prj_valid_align_i)

        return np.array(gtj_valid), np.array(prj_valid_align)


def visual_align(gtj0, prj0, key):
    gtj = gtj0.copy()
    prj = prj0.copy()

    if key in ["stb", "rhd", "ah"]:
        # gtj :B*21*3
        # prj :B*21*3
        root_idx = 0  # root
        ref_bone_link = [0, 9]  # mid mcp
        pred_align = prj.copy()
        for i in range(prj.shape[0]):

            pred_ref_bone_len = np.linalg.norm(prj[i][ref_bone_link[0]] - prj[i][ref_bone_link[1]])
            gt_ref_bone_len = np.linalg.norm(gtj[i][ref_bone_link[0]] - gtj[i][ref_bone_link[1]])
            scale = gt_ref_bone_len / pred_ref_bone_len
            # scale = 0.0824 / pred_ref_bone_len

            for j in range(21):
                pred_align[i][j] = gtj[i][root_idx] + scale * (prj[i][j] - prj[i][root_idx])

        # gtj -= gtj0[:, root_idx: root_idx + 1, :]
        # pred_align -= pred_align[:, root_idx: root_idx + 1, :]

        return gtj, pred_align


def global_norm_align(gtj0, prj0, key):
    gtj = gtj0.copy()
    prj = prj0.copy()

    if key in ["stb", "rhd", "ah"]:
        # gtj :B*21*3
        # prj :B*21*3
        root_idx = 20  # root
        ref_bone_link = [0, 9]  # mid mcp
        pred_norm_align = prj.copy()
        gt_norm = gtj.copy()
        for i in range(prj.shape[0]):

            pred_ref_bone_len = np.linalg.norm(prj[i][ref_bone_link[0]] - prj[i][ref_bone_link[1]])
            gt_ref_bone_len = np.linalg.norm(gtj[i][ref_bone_link[0]] - gtj[i][ref_bone_link[1]])

            for j in range(21):
                pred_norm_align[i][j] = 1 / pred_ref_bone_len * (prj[i][j] - prj[i][root_idx])
                gt_norm[i][j] = 1 / gt_ref_bone_len * (gtj[i][j] - gtj[i][root_idx])

        return gt_norm, pred_norm_align
