from torchvision.transforms.functional import *
import config as cfg
from scipy.optimize import minimize


def batch_denormalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out_testset of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError('invalid tensor or tensor channel is not BCHW')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[None, :, None, None]).sub_(-1 * mean[None, :, None, None])
    return tensor


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    else:
        return tensor


def bhwc_2_bchw(tensor):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError('invalid tensor or tensor channel is not BCHW')
    return tensor.unsqueeze(1).transpose(1, -1).squeeze(-1)


def bchw_2_bhwc(tensor):
    """
    :param x: torch tensor, B x C x H x W
    :return:  torch tensor, B x H x W x C
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError('invalid tensor or tensor channel is not BCHW')
    return tensor.unsqueeze(-1).transpose(1, -1).squeeze(1)


def initiate(label=None):
    if label == "zero":
        shape = torch.zeros(10).unsqueeze(0)
        pose = torch.zeros(48).unsqueeze(0)
    elif label == "uniform":
        shape = torch.from_numpy(np.random.normal(size=[1, 10])).float()
        pose = torch.from_numpy(np.random.normal(size=[1, 48])).float()
    elif label == "01":
        shape = torch.rand(1, 10)
        pose = torch.rand(1, 48)
    else:
        raise ValueError("{} not in ['zero'|'uniform'|'01']".format(label))
    return pose, shape


def cross_merge_two_vec(vec1, vec2):
    assert vec1.shape == vec2.shape, "Two vecs shape not matched."
    bs, *shape = vec1.shape
    merged = np.empty((bs * 2, *shape))
    merged[::2] = vec1
    merged[1::2] = vec2
    return merged


def cross_merge_two_list(ls1, ls2):
    assert len(ls1) == len(ls2), "Two lists shape not matched."
    merged = [val for pair in zip(ls1, ls2) for val in pair]
    return merged


def R_from_2poses(joints0, joints1, is_torch=False):
    '''
    input: joints0 & joints1
    output: rotation matrix R, CS0->CS1
    '''
    if is_torch:
        joints0, joints1 = torch.stack(joints0), torch.stack(joints1)
        Cov = joints0.T @ joints1
        U, S, V = torch.svd(Cov)
        return U @ V.T
    else:
        joints0, joints1 = np.array(joints0), np.array(joints1)
        Cov = joints0.T @ joints1
        U, S, Vt = np.linalg.svd(Cov)
        return U @ Vt


# Used only for calc_metrics.py
def merge_from_2hands(hands1, hands2, valids1, valids2, R12=None, R21=None):
    if isinstance(hands1, torch.Tensor):
        hands1 = hands1.detach().cpu().numpy()
    if isinstance(hands2, torch.Tensor):
        hands2 = hands2.detach().cpu().numpy()
    if isinstance(valids1, torch.Tensor):
        valids1 = valids1.detach().cpu().numpy()
    if isinstance(valids2, torch.Tensor):
        valids2 = valids2.detach().cpu().numpy()
    if isinstance(R12, torch.Tensor):
        R12 = R12.detach().cpu().numpy()
    if isinstance(R21, torch.Tensor):
        R21 = R21.detach().cpu().numpy()

    pseudo1, pseudo2 = [], []
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

        # print(pred1.shape, (pred2 @ R21).shape)
        preds1 = np.array([pred1, pred2 @ R21])
        preds2 = np.array([pred1 @ R12, pred2])
        v = np.array([valids1[i], valids2[i]])
        v = v[:, :, np.newaxis]

        pseudo1.append(avg_pred(preds1, v))
        pseudo2.append(avg_pred(preds2, v))

    return np.array(pseudo1), np.array(pseudo2)


def pseudo_from_2hands(hands1, hands2, valids1, valids2, vmax1, vmax2, R12=None, R21=None, merge='weight'):
    if isinstance(hands1, torch.Tensor):
        hands1 = hands1.detach().cpu().numpy()
    if isinstance(hands2, torch.Tensor):
        hands2 = hands2.detach().cpu().numpy()
    if isinstance(valids1, torch.Tensor):
        valids1 = valids1.detach().cpu().numpy()
    if isinstance(valids2, torch.Tensor):
        valids2 = valids2.detach().cpu().numpy()
    if isinstance(R12, torch.Tensor):
        R12 = R12.detach().cpu().numpy()
    if isinstance(R21, torch.Tensor):
        R21 = R21.detach().cpu().numpy()

    pseudo1, pseudo2 = [], []
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

        # print(pred1.shape, (pred2 @ R21).shape)
        preds1 = np.array([pred1, pred2 @ R21])
        preds2 = np.array([pred1 @ R12, pred2])
        v = np.array([valids1[i], valids2[i]])
        v = v[:, :, np.newaxis]
        vmx = np.array([vmax1[i], vmax2[i]])

        avg_op = weighted_avg_pred
        if merge == 'avg':
            avg_op = avg_pred
        if merge == 'max':
            avg_op = max_avg_pred

        pseudo1.append(avg_op(preds1, v, vmx))
        pseudo2.append(avg_op(preds2, v, vmx))

    return np.array(pseudo1), np.array(pseudo2)


def avg_pred(preds, valids, vmax=None):
    assert len(preds) == len(valids)

    for i in range(valids.shape[1]):
        if np.sum(valids[:, i]) == 0:
            valids[:, i] = 1
        valids[:, i] /= np.sum(valids[:, i])

    preds *= valids

    avg = np.sum(preds, axis=0)
    return avg


def weighted_avg_pred(preds, valids, vmax):
    assert len(preds) == len(valids)

    for i in range(valids.shape[1]):
        if np.sum(valids[:, i]) == 0:
            valids[:, i] = 1
    valids[valids == 0] = -np.inf

    weight = valids * vmax
    weight = np.exp(weight)
    weight /= np.sum(weight, axis=0, keepdims=True)

    preds *= weight

    avg = np.sum(preds, axis=0)
    return avg


def max_avg_pred(preds, valids, vmax):
    assert len(preds) == len(valids)

    for i in range(valids.shape[1]):
        if np.sum(valids[:, i]) == 0:
            valids[:, i] = 1

    vmax = valids * vmax
    max_values = np.max(vmax, axis=0)
    weight = vmax == max_values

    preds *= weight

    avg = np.sum(preds, axis=0)
    return avg


def quaternion_from_matrix(R, is_torch=False):
    if is_torch:
        q = torch.empty(4, dtype=R.dtype, device=R.device)
        tr = torch.trace(R)
    else:
        q = np.empty((4,))
        tr = np.trace(R)
    sqrt_fun = torch.sqrt if is_torch else np.sqrt

    if tr > 0:
        S = sqrt_fun(tr + 1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = sqrt_fun(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = sqrt_fun(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = sqrt_fun(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S

    return q


def matrix_from_quaternion(q, is_torch=False):
    if is_torch:
        R = torch.empty((3, 3), dtype=q.dtype, device=q.device)
        q = q / torch.linalg.norm(q)
    else:
        R = np.empty((3, 3))
        q = q / np.linalg.norm(q)  # Normalize the quaternion if necessary

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
    q0q1, q0q2, q0q3 = q0 * q1, q0 * q2, q0 * q3
    q1q2, q1q3, q2q3 = q1 * q2, q1 * q3, q2 * q3

    R[0, 0] = q0q0 + q1q1 - q2q2 - q3q3
    R[0, 1] = 2 * (q1q2 - q0q3)
    R[0, 2] = 2 * (q1q3 + q0q2)

    R[1, 0] = 2 * (q1q2 + q0q3)
    R[1, 1] = q0q0 - q1q1 + q2q2 - q3q3
    R[1, 2] = 2 * (q2q3 - q0q1)

    R[2, 0] = 2 * (q1q3 - q0q2)
    R[2, 1] = 2 * (q2q3 + q0q1)
    R[2, 2] = q0q0 - q1q1 - q2q2 + q3q3

    return R


def avg_R_from_2poses(hands1, hands2, valids1, valids2, is_torch=False, quatn_mo=None, thres=0.4):
    Rs = []
    for i in range(hands1.shape[0]):
        pred1, pred2 = hands1[i], hands2[i]
        p1, p2 = [], []
        for j in range(pred1.shape[0]):
            if valids1[i, j] and valids2[i, j]:
                p1.append(pred1[j])
                p2.append(pred2[j])
        Rs.append(R_from_2poses(p1, p2, is_torch=is_torch))

    quatns = [quaternion_from_matrix(R, is_torch=is_torch) for R in Rs]
    if is_torch:
        quatn_avg_raw = torch.mean(torch.stack(quatns), dim=0)
        quatn_avg = quatn_avg_raw / torch.linalg.norm(quatn_avg_raw)
    else:
        quatn_avg = np.mean(quatns, axis=0)
        quatn_avg /= np.linalg.norm(quatn_avg)
    R_avg = matrix_from_quaternion(quatn_avg, is_torch=is_torch)

    return R_avg, quatn_avg


def update_R(R0, R1, momentum=0.99, is_torch=False):
    if is_torch:
        q0 = quaternion_from_matrix(R0.detach(), is_torch=is_torch)
        q1 = quaternion_from_matrix(R1.detach(), is_torch=is_torch)
    else:
        q0 = quaternion_from_matrix(R0, is_torch=is_torch)
        q1 = quaternion_from_matrix(R1, is_torch=is_torch)
    q = q0 * momentum + (1 - momentum) * q1
    R_updated = matrix_from_quaternion(q, is_torch=is_torch)
    return R_updated


def joint_to_locmap_deltamap(joint, joint_root_idx=0, ref_bone_link=(0, 9)):
    joint_bone = 0
    for jid, nextjid in zip(ref_bone_link[:-1], ref_bone_link[1:]):
        joint_bone += np.linalg.norm(joint[nextjid] - joint[jid])
    joint_root = joint[joint_root_idx]
    joint_bone = np.atleast_1d(joint_bone)

    '''prepare location maps L'''
    jointR = joint - joint_root[np.newaxis, :]  # root relative
    jointRS = jointR / joint_bone  # scale invariant
    # '''jointRS.shape= (21, 3) to locationmap(21,3,32,32)'''
    location_map = jointRS[:, :, np.newaxis, np.newaxis].repeat(32, axis=-2).repeat(32, axis=-1)

    '''prepare delta maps D'''
    kin_chain = [
        jointRS[i] - jointRS[cfg.SNAP_PARENT[i]]
        for i in range(21)
    ]
    kin_chain = np.array(kin_chain)  # id 0's parent is itself #21*3
    kin_len = np.linalg.norm(
        kin_chain, ord=2, axis=-1, keepdims=True  # 21*1
    )
    kin_chain[1:] = kin_chain[1:] / kin_len[1:]
    # '''kin_chain(21, 3) to delta_map(21,3,32,32)'''
    delta_map = kin_chain[:, :, np.newaxis, np.newaxis].repeat(32, axis=-2).repeat(32, axis=-1)

    return location_map, delta_map


def objective(x, R0):
    # Define the objective function to minimize
    l = len(x)
    j0, j1 = np.array(x[:l // 2]).reshape(-1, 3), np.array(x[l // 2:]).reshape(-1, 3)
    R = R_from_2poses(j0, j1)
    error = np.linalg.norm(R - R0, 'fro')
    # print(error)
    return error


def stb_from_2hands(hands1, hands2, valids1, valids2, R0):
    stable1, stable2 = [], []
    for i in range(hands1.shape[0]):
        pred1, pred2 = hands1[i], hands2[i]
        p1, p2 = [], []
        for j in range(pred1.shape[0]):
            if valids1[i, j] and valids2[i, j]:
                p1.append(pred1[j])
                p2.append(pred2[j])

        x_in = np.array([p1, p2])
        results = minimize(objective, x_in, args=R0, method='BFGS', options={'maxiter': 20})
        opt_pred = results.x
        # print(f'======{results.fun}======')
        l = len(opt_pred)
        j1, j2 = np.array(opt_pred[:l // 2]).reshape(-1, 3), np.array(opt_pred[l // 2:]).reshape(-1, 3)

        joints1, joints2 = [], []
        idx = 0
        for j in range(pred1.shape[0]):
            if valids1[i, j] and valids2[i, j]:
                joints1.append(j1[idx])
                joints2.append(j2[idx])
                idx += 1
            else:
                joints1.append(pred1[j])
                joints2.append(pred2[j])
        stable1.append(joints1)
        stable2.append(joints2)
    return np.array(stable1), np.array(stable2)


if __name__ == '__main__':
    a = np.ones((4, 2, 2))
    b = a * 2
    print(cross_merge_two_vec(b, a))
