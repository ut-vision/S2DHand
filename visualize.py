import glob
import json
import os.path

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import func
from datasets.assemblyhands.utils.transforms import cam2pixel

connections = {
    'pinky': [0, 17, 18, 19, 20],
    'ring': [0, 13, 14, 15, 16],
    'middle': [0, 9, 10, 11, 12],
    'index': [0, 5, 6, 7, 8],
    'thumb': [0, 1, 2, 3, 4],
}

# Define unique colors for each finger in RGB format
colors = {
    'thumb': (153, 0, 255),
    'index': (0, 207, 58),
    'middle': (255, 0, 64),
    'ring': (255, 223, 0),
    'pinky': (0, 123, 255)
}


def visualize_3d_joints(idx, checkpoint, eid=37, setup='set0', pair='0,1', is_gt=False):
    JNUM = 21
    R_config = json.load(open('./R_config.json'))
    R12 = np.array(R_config[f'set{setup}-{pair}']['R_gt'])

    crop_root = './data/assemblyhands_crop/images/ego_images_rectified/test/'
    log = os.path.join(checkpoint, f'{eid}-set{setup}-{pair}.log')
    lines = open(log).readlines()
    lines.pop(0)
    mono = float(lines.pop(-1).split(':')[-1])
    line_num = len(lines)

    imgs, joints, fs, ps, valids = [], [], [], [], []
    for l in range(idx * 2, idx * 2 + 2):
        line = lines[l].split()
        name = line[0].replace('.jpg', f'_{line[11]}.jpg')
        name = os.path.join(crop_root, name)
        cam = name.split('/')[-2]
        name = name.replace(cam, f'{cam}_mono10bit')
        print(name)
        img = cv2.imread(name)

        gt = np.array(line[5].split(',')).astype(np.float64).reshape(JNUM, 3)
        pred = np.array(line[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        visual_joints = gt if is_gt else pred

        focal = np.array(line[9].split(',')).astype(np.float64)
        pricpt = np.array(line[10].split(',')).astype(np.float64)
        valid = np.array(line[6].split(',')).astype(np.float64).reshape(JNUM, )

        imgs.append(img)
        joints.append(visual_joints)
        fs.append(focal)
        ps.append(pricpt)
        valids.append(valid)

    if eid not in [37, 68]:  # These are the index of pre-trained models
        anchors = [joints[k][0, :] for k in range(len(joints))]
        joint0_align = joints[0] - joints[0][0, :]
        joint1_align = joints[1] - joints[1][0, :]
        merge0, merge1 = func.merge_from_2hands(joint0_align[np.newaxis, :, :], joint1_align[np.newaxis, :, :],
                                                valids[0][np.newaxis, :], valids[1][np.newaxis, :], R12=R12, R21=R12.T)
        merge0, merge1 = merge0.squeeze(0), merge1.squeeze(0)
        joints = [merge0 + anchors[0], merge1 + anchors[1]]

    for i in range(2):
        img, visual_joints, focal, pricpt = imgs[i], joints[i], fs[i], ps[i]
        kp = cam2pixel(visual_joints, focal, pricpt).astype(np.int32)
        x_points = kp[:, 0]
        y_points = kp[:, 1]
        x_points = x_points.astype(np.int32)
        y_points = y_points.astype(np.int32)

        # Draw points onto the image
        radius = 5
        color = (0, 0, 255)  # Red in BGR format
        thickness = -1  # Fill the circle

        for finger, conn in connections.items():
            for i in range(len(conn) - 1):
                cv2.line(img, (kp[conn[i], 0], kp[conn[i], 1]),
                         (kp[conn[i + 1], 0], kp[conn[i + 1], 1]),
                         colors[finger], 5)
        for x, y in zip(x_points, y_points):
            cv2.circle(img, (x, y), radius, color, thickness)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image using matplotlib
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Train: DetNet')
    # Dataset setting
    parser.add_argument('--checkpoint', type=str, default='visualize/evaluation/ah',
                        help='save dir of the test logs')
    parser.add_argument('--setup', type=int, default=0, help='id of headset')
    parser.add_argument('--pair', type=str, default='1,2', help='id of dual-camera pair')
    parser.add_argument('-eid', '--evaluate_id', default=37, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--sample_idx', type=int, default=18, help='id of visualized sample')
    args = parser.parse_args()

    sample_idx = args.sample_idx
    setup, pair = args.setup, args.pair
    visualize_3d_joints(sample_idx, args.checkpoint, eid=args.evaluate_id, setup=setup, pair=pair)
    visualize_3d_joints(sample_idx, args.checkpoint, eid=args.evaluate_id, setup=setup, pair=pair, is_gt=True)
