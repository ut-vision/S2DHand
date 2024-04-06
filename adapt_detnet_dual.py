import argparse
import os


def parse_the_args():
    parser = argparse.ArgumentParser(description='PyTorch Train: DetNet')
    # Dataset setting
    parser.add_argument('--gpus', type=str, default="0", help='gpu ids')
    parser.add_argument('-dr', '--data_root', type=str, default="data", help='dataset root directory')
    parser.add_argument("-trs", "--datasets_train", nargs="+", default=['cmu', 'rhd'], type=str,
                        help="sub datasets, should be listed in: [cmu|rhd|gan]")
    parser.add_argument("-tes", "--datasets_test", nargs="+", default=['rhd', 'stb', "do", "eo"], type=str,
                        help="sub datasets, should be listed in: [rhd|stb|do|eo]")
    # Miscs
    parser.add_argument('-ckp', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-sp', '--saved_prefix', default='ckp_detnet', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-op', '--outpath', default='out_loss_auc', type=str, metavar='PATH',
                        help='path to out_testset loss and auc (default: out_testset)')
    parser.add_argument('--snapshot', default=1, type=int, help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                        help='whether to load checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # Training Parameters
    parser.add_argument('-eid', '--evaluate_id', default=319, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--pic', default=-1, type=int, metavar='N', help='number of input images')
    parser.add_argument('-c', '--clean', dest='clean', action='store_true',
                        help='clean model on one gpu if trained on 2 gpus')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-se', '--start_epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--train_batch', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('-tb', '--test_batch', default=512, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-lr', '--learning-rate', default=0.0005, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument("--lr_decay_step", default=250, type=int, help="Epochs after which to decay learning rate", )
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--alpha', type=float, default=0.7, help='the weight of consist and stable pseudo.')
    parser.add_argument('--avg', type=str, default='max', help='the way to generate pseudo labels.')
    parser.add_argument('--det_loss', dest='det_loss', action='store_true', help='Calculate detnet loss', default=True)
    parser.add_argument('--setup', type=int, default=0, help='id of headset')
    parser.add_argument('--pair', type=str, default='1,2', help='id of dual-camera pair')
    parser.add_argument('--initR', type=str, default='pred', help='ways to init quatn, pred or gt')
    parser.add_argument('--root_idx', type=int, default=0, help='root joint id for alignment')

    parse_args = parser.parse_args()
    return parse_args


parse_args = parse_the_args()
os.environ["CUDA_VISIBLE_DEVICES"] = parse_args.gpus

import time
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import losses.detloss as losses
import utils.misc as misc
from datasets.egodexter import EgoDexter
from datasets.handataset import HandDataset
from model.detnet.detnet import detnet
from utils import func, align
from utils.eval.evalutils import AverageMeter, accuracy_heatmap
from utils.eval.zimeval import EvalUtil
from utils import vis
import random

# select proper device to run
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
DEBUG = 0


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def updata_ema_params(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def main(args):
    for path in [args.checkpoint, args.outpath]:
        if not os.path.isdir(path):
            os.makedirs(path)

    misc.print_args(args)

    print("\nCREATE NETWORK")
    model = detnet()
    model.to(device)
    if args.resume:
        model_ema = detnet()
        model_ema.to(device)

    # define loss function (criterion) and optimizer
    criterion_det = losses.DetLoss(
        lambda_hm=100.,
        lambda_dm=1.,
        lambda_lm=10.,
    )
    criterion = {
        'det': criterion_det
    }
    optimizer = torch.optim.Adam(
        [
            {
                'params': model.parameters(),
                'initial_lr': args.learning_rate
            },

        ],
        lr=args.learning_rate
    )

    test_set_dic = {}
    test_loader_dic = {}
    best_acc = {}
    auc_all = {}
    acc_hm_all = {}
    for test_set_name in args.datasets_test:
        if test_set_name in ['stb', 'rhd', 'do', 'ah']:
            test_set_dic[test_set_name] = HandDataset(
                data_split='test',
                train=False,
                subset_name=test_set_name,
                data_root=args.data_root,
                pic=parse_args.pic if args.evaluate else -1,
                setup=args.setup, pair=tuple(map(int, args.pair.split(','))),
                root_idx=args.root_idx
            )
        elif test_set_name == 'eo':
            test_set_dic[test_set_name] = EgoDexter(
                data_split='test',
                data_root=args.data_root,
                hand_side="right"
            )
            print(test_set_dic[test_set_name])

        test_loader_dic[test_set_name] = torch.utils.data.DataLoader(
            test_set_dic[test_set_name],
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True, drop_last=False
        )
        best_acc[test_set_name] = 0
        auc_all[test_set_name] = []
        acc_hm_all[test_set_name] = []
    # 1 calculate the length of the test set
    total_test_set_size = 0
    for key, value in test_set_dic.items():
        total_test_set_size += len(value)
    print("Total test set size: {}".format(total_test_set_size))

    # 2 load/initialize the parameters
    if args.evaluate:
        loadpath = os.path.join(args.checkpoint, 'ckp_detnet_{}.pth'.format(args.evaluate_id))
        print("\nLOAD CHECKPOINT", loadpath)
        state_dict = torch.load(loadpath)
        # if args.clean:
        state_dict = misc.clean_state_dict(state_dict)

        model.load_state_dict(state_dict)
    elif args.resume:
        loadpath = os.path.join(args.checkpoint, 'ckp_detnet_{}.pth'.format(args.evaluate_id))
        print("\nLOAD CHECKPOINT", loadpath)
        try:
            state_dict = torch.load(loadpath)
        except:
            loadpath = os.path.join('pretrain', 'ckp_detnet_{}.pth'.format(args.evaluate_id))
            print("CHECKPOINT NOT EXIST, TRY", loadpath)
            state_dict = torch.load(loadpath)
        # if args.clean:
        state_dict = misc.clean_state_dict(state_dict)

        model.load_state_dict(state_dict)
        model_ema.load_state_dict(state_dict)
        for param in model_ema.parameters():
            param.detach_()
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
    if args.initR is None:
        R_mom = None
    else:
        R_config = json.load(open('R_config.json'))
        R_mom = np.array(R_config[f'set{args.setup}-{args.pair}'][f'R_{args.initR}'])

    # 3 Validation branch
    if args.evaluate:
        for key, value in test_loader_dic.items():
            validate(value, model, criterion, key, epoch=None, args=args)
        return

    # 4 Training branch
    train_dataset = HandDataset(
        data_split='test',
        train=False,
        subset_name=args.datasets_train,
        data_root=args.data_root,
        scale_jittering=0.1,
        center_jettering=0.1,
        max_rot=0.5 * np.pi,
        pic=parse_args.pic,
        setup=args.setup, pair=tuple(map(int, args.pair.split(','))),
        root_idx=args.root_idx
    )

    print("Total train dataset size: {}".format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    # DataParallel so u can use multi GPUs
    model = torch.nn.DataParallel(model)
    if args.resume:
        model_ema = torch.nn.DataParallel(model_ema)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    acc_hm = {}
    loss_all = {"lossH": [],
                "lossD": [],
                "lossL": [],
                }

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print(f'\nEpoch: {epoch}/{args.start_epoch + args.epochs - 1}')
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  trian for one epoch  ###############
        train(
            train_loader,
            model,
            model_ema,
            criterion,
            optimizer,
            args=args, loss_all=loss_all, epoch=epoch, R12=R_mom
        )
        ##################################################
        auc = best_acc.copy()  # need to deepcopy it because it's a dict
        for key, value in test_loader_dic.items():
            auc[key], acc_hm[key] = validate(value, model, criterion, key, epoch=epoch, args=args, write_epoch=epoch)
            auc_all[key].append([epoch, auc[key]])
            acc_hm_all[key].append([epoch, acc_hm[key]])

        misc.save_checkpoint(
            {
                'epoch': epoch,
                'model': model,
            },
            checkpoint=args.checkpoint,
            filename='{}.pth'.format(args.saved_prefix),
            snapshot=args.snapshot,
            is_best=[auc, best_acc]
        )

        for key, value in test_loader_dic.items():
            if auc[key] > best_acc[key]:
                best_acc[key] = auc[key]

        misc.out_loss_auc(loss_all, auc_all, acc_hm_all, outpath=args.outpath)

        scheduler.step()

    return  # end of main


def one_forward_pass(metas, model, criterion, args, train=True):
    clr = metas['clr'].to(device, non_blocking=True)

    ''' prepare infos '''
    if 'hm_veil' in metas.keys():
        hm_veil = metas['hm_veil'].to(device, non_blocking=True)  # (B,21)

        infos = {
            'hm_veil': hm_veil,
            'batch_size': clr.shape[0]
        }

        ''' prepare targets '''

        hm = metas['hm'].to(device, non_blocking=True)
        delta_map = metas['delta_map'].to(device, non_blocking=True)
        location_map = metas['location_map'].to(device, non_blocking=True)
        flag_3d = metas['flag_3d'].to(device, non_blocking=True)
        joint = metas['joint'].to(device, non_blocking=True)

        targets = {
            'clr': clr,
            'hm': hm,
            'dm': delta_map,
            'lm': location_map,
            "flag_3d": flag_3d,
            "joint": joint,
            "uv": metas['kp2d'],
            "hand": metas['hand_type']
        }
        if args.evaluate:
            eval_targets = {
                'cam_param': metas['cam_param'],
                "seq": metas['seq_name'],
                'cam': metas['cam_name'],
                'frame': metas['frame']
            }
            targets.update(eval_targets)
    else:
        infos = {
            'batch_size': clr.shape[0]
        }
        tips = metas['tips'].to(device, non_blocking=True)
        targets = {
            'clr': clr,
            "joint": tips

        }

    ''' ----------------  Forward Pass  ---------------- '''
    results = model(clr)
    left_mapping = {'left': 1, 'right': 0}
    left_mask = torch.tensor([left_mapping[hand] for hand in metas['hand_type']])
    results['xyz'][left_mask == 1, :, 0] *= -1  # flip left hand back
    results['left_mask'] = left_mask
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).cuda()
    losses = {}

    if not train:
        return results, {**targets, **infos}, total_loss, losses

    # ''' compute losses '''
    # if args.det_loss:
    #     det_total_loss, det_losses, batch_3d_size = criterion['det'].compute_loss(
    #         results, targets, infos
    #     )
    #     total_loss += det_total_loss
    #     losses.update(det_losses)
    #
    #     targets["batch_3d_size"] = batch_3d_size
    #
    # return results, {**targets, **infos}, total_loss, losses


def validate(val_loader, model, criterion, key, epoch, args, stop=-1, write_epoch=None):
    print("{}_test_set under test".format(key))
    # switch to evaluate mode
    model.eval()

    if key in ["stb", "rhd", "ah"]:
        am_accH = AverageMeter()

    evaluator = EvalUtil()

    if args.evaluate or write_epoch:
        count = 0
        logpath = os.path.join(args.checkpoint, 'evaluation', key,
                               f'{write_epoch if write_epoch else args.evaluate_id}-set{args.setup}-{args.pair}.log')
        if not os.path.exists(os.path.dirname(logpath)):
            os.makedirs(os.path.dirname(logpath))
        fw = open(logpath, 'w')
        fw.write(f'img seq cam frame pred gt valid camrot campos focal pricpt hand handpred abs_depth depth_info uv\n')

    with torch.no_grad():
        for i, (metas1, metas2) in tqdm(enumerate(val_loader)):
            preds1, targets1, _1, _2 = one_forward_pass(
                metas1, model, criterion, args=args, train=False
            )
            preds2, targets2, _1, _2 = one_forward_pass(
                metas2, model, criterion, args=args, train=False
            )

            if key in ["stb", "rhd", "ah"]:
                # heatmap accuracy
                avg_acc_hm, _ = accuracy_heatmap(
                    preds1['h_map'],
                    targets1['hm'],
                    targets1['hm_veil']
                )
                am_accH.update(avg_acc_hm, targets1['batch_size'])

            pred_joint1 = func.to_numpy(preds1['xyz'])
            pred_joint2 = func.to_numpy(preds2['xyz'])

            gt_joint1 = func.to_numpy(targets1['joint'])
            gt_joint2 = func.to_numpy(targets2['joint'])

            gt_joint1, pred_joint_align1 = align.global_align(gt_joint1, pred_joint1, key=key, root_idx=args.root_idx)
            gt_joint2, pred_joint_align2 = align.global_align(gt_joint2, pred_joint2, key=key, root_idx=args.root_idx)
            valid1, valid2 = metas1['vis'].numpy(), metas2['vis'].numpy()
            gt_joint1 *= 1000.
            pred_joint_align1 *= 1000
            gt_joint2 *= 1000.
            pred_joint_align2 *= 1000

            if 'vis' in metas1.keys():
                for targj, predj_a, kp_vis in zip(gt_joint1, pred_joint_align1, valid1):
                    evaluator.feed(targj, predj_a, keypoint_vis=kp_vis)
            else:
                for targj, predj_a in zip(gt_joint1, pred_joint_align1):
                    evaluator.feed(targj, predj_a)
            if 'vis' in metas2.keys():
                for targj, predj_a, kp_vis in zip(gt_joint2, pred_joint_align2, valid2):
                    evaluator.feed(targj, predj_a, keypoint_vis=kp_vis)
            else:
                for targj, predj_a in zip(gt_joint2, pred_joint_align2):
                    evaluator.feed(targj, predj_a)

            # Write the evaluation log, the code is a bit too long.
            if args.evaluate or write_epoch:
                pred_joint_align = func.cross_merge_two_vec(pred_joint_align1, pred_joint_align2)
                gt_joint = func.cross_merge_two_vec(gt_joint1, gt_joint2)
                pred1d, gt1d = pred_joint_align.reshape(-1, 21 * 3), gt_joint.reshape(-1, 21 * 3)

                bs = len(pred_joint_align1)
                valid = func.cross_merge_two_vec(valid1, valid2)

                fcl1, prcpt1 = metas1['cam_param']['focal'], metas1['cam_param']['princpt']
                focal1d1 = [[fcl1[0][f].numpy(), fcl1[1][f].numpy()] for f in range(bs)]
                fcl2, prcpt2 = metas2['cam_param']['focal'], metas2['cam_param']['princpt']
                focal1d2 = [[fcl2[0][f].numpy(), fcl2[1][f].numpy()] for f in range(bs)]
                focal1d = func.cross_merge_two_list(focal1d1, focal1d2)

                princpt1d1 = [[prcpt1[0][f].numpy(), prcpt1[1][f].numpy()] for f in range(bs)]
                princpt1d2 = [[prcpt2[0][f].numpy(), prcpt2[1][f].numpy()] for f in range(bs)]
                princpt1d = func.cross_merge_two_list(princpt1d1, princpt1d2)

                rot1d1 = [f.numpy().reshape(-1) for f in metas1['cam_param']['rot']]
                rot1d2 = [f.numpy().reshape(-1) for f in metas2['cam_param']['rot']]
                rot1d = func.cross_merge_two_list(rot1d1, rot1d2)

                pos1d1 = [f.numpy().reshape(-1) for f in metas1['cam_param']['pos']]
                pos1d2 = [f.numpy().reshape(-1) for f in metas2['cam_param']['pos']]
                pos1d = func.cross_merge_two_list(pos1d1, pos1d2)

                frame_ls = func.cross_merge_two_list(metas1['frame'], metas2['frame'])
                seq_ls = func.cross_merge_two_list(metas1['seq_name'], metas2['seq_name'])
                cam_ls = func.cross_merge_two_list(metas1['cam_name'], metas2['cam_name'])
                hand_ls = func.cross_merge_two_list(targets1['hand'], targets2['hand'])

                for l in range(bs * 2):
                    frame, seq, cam, hand = frame_ls[l], seq_ls[l], cam_ls[l], hand_ls[l]
                    img = f'{seq}/{cam}/{frame:06d}.jpg'

                    p = ','.join([str(u) for u in pred1d[l]])
                    g = ','.join([str(u) for u in gt1d[l]])
                    v = ','.join([str(u) for u in valid[l]])
                    rot = ','.join([str(u) for u in rot1d[l]])
                    pos = ','.join([str(u) for u in pos1d[l]])
                    fo = ','.join([str(u) for u in focal1d[l]])
                    pr = ','.join([str(u) for u in princpt1d[l]])
                    hpred, ab, d = '0', '0', '0'
                    # uv = ','.join([str(u) for u in uv1d[l]])

                    fw.write(' '.join(
                        [img, seq, cam, str(frame.numpy()), p, g, v, rot, pos, fo, pr, hand, hpred, ab, d]) + '\n')
                    count += 1

            if stop != -1 and i >= stop:
                break

    (
        _1, mean_all, _3,
        auc_all,
        pck_curve_all,
        thresholds
    ) = evaluator.get_measures(
        20, 50, 15
    )
    print("AUC all of {}_test_set is : {}".format(key, auc_all))
    print("Mean all of {}_test_set is : {}".format(key, mean_all))
    print(f"EPE: {np.mean(mean_all)}")

    if args.evaluate or write_epoch:
        print(f'{count} samples written into {logpath}.')
        fw.write(f"AUC: {auc_all}, EPE: {np.mean(mean_all)}")
        fw.close()

    if key in ["stb", "rhd", "ah"]:
        return auc_all, am_accH.avg
    elif key in ["do", "eo"]:
        return auc_all, 0


def train(train_loader, model, model_ema, criterion, optimizer, args, loss_all, epoch=0, R12=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    am_loss_hm = AverageMeter()
    am_loss_dm = AverageMeter()
    am_loss_lm = AverageMeter()

    last = time.time()
    # switch to train
    model.eval()
    model_ema.eval()
    itr_num = len(train_loader)
    # bar = Bar('\033[31m Train \033[0m')
    # for i, metas in tqdm(enumerate(train_loader)):
    for i, (metas1, metas2) in enumerate(train_loader):
        data_time.update(time.time() - last)
        # results, targets, total_loss, losses = one_forward_pass(
        #     metas, model, criterion, args, train=True
        # )
        preds1, targets1, _1, _2 = one_forward_pass(
            metas1, model, criterion, args=args, train=False
        )
        preds2, targets2, _1, _2 = one_forward_pass(
            metas2, model, criterion, args=args, train=False
        )
        preds1_ema, targets1_ema, _1, _2 = one_forward_pass(
            metas1, model_ema, criterion, args=args, train=False
        )
        preds2_ema, targets2_ema, _1, _2 = one_forward_pass(
            metas2, model_ema, criterion, args=args, train=False
        )

        pred_e1 = func.to_numpy(preds1_ema['xyz'])
        pred_e2 = func.to_numpy(preds2_ema['xyz'])
        vmax_e1 = func.to_numpy(preds1_ema['vmax'])
        vmax_e2 = func.to_numpy(preds2_ema['vmax'])
        # pred_j1 = func.to_numpy(preds1['xyz'])
        # pred_j2 = func.to_numpy(preds2['xyz'])
        bs = pred_e1.shape[0]

        '''calculate pseudo-labels'''
        valid1, valid2 = metas1['vis'].numpy(), metas2['vis'].numpy()
        pred_e1_align, pred_e2_align, pred_e1_anchor,\
            pred_e2_anchor, scale_e2 = align.align_two_pred(pred_e1, pred_e2, root_idx=args.root_idx)

        consis1_align, consis2_align = func.pseudo_from_2hands(pred_e1_align, pred_e2_align, valid1, valid2,
                                                               vmax_e1, vmax_e2, R12=R12, R21=R12.T, merge=args.avg)
        stable1_align, stable2_align = func.stb_from_2hands(pred_e1_align, pred_e2_align, valid1, valid2, R12)

        consist1 = consis1_align + pred_e1_anchor
        consist2 = consis2_align / scale_e2 + pred_e2_anchor

        stable1 = stable1_align + pred_e1_anchor
        stable2 = stable2_align / scale_e2 + pred_e2_anchor

        pseudo1 = args.alpha * consist1 + (1 - args.alpha) * stable1
        pseudo2 = args.alpha * consist2 + (1 - args.alpha) * stable2

        R, _ = func.avg_R_from_2poses(pseudo1, pseudo2, valid1, valid2)

        pseudo1[preds1_ema['left_mask'] == 1, :, 0] *= -1
        pseudo2[preds2_ema['left_mask'] == 1, :, 0] *= -1

        '''generate hmap, dmap, and lmap for supervision'''
        lmap1, dmap1 = np.empty((bs, 21, 3, 32, 32)), np.empty((bs, 21, 3, 32, 32))
        lmap2, dmap2 = np.empty((bs, 21, 3, 32, 32)), np.empty((bs, 21, 3, 32, 32))
        for k in range(bs):
            l1, d1 = func.joint_to_locmap_deltamap(pseudo1[k], train_loader.dataset.joint_root_idx,
                                                   train_loader.dataset.ref_bone_link)
            l2, d2 = func.joint_to_locmap_deltamap(pseudo2[k], train_loader.dataset.joint_root_idx,
                                                   train_loader.dataset.ref_bone_link)
            lmap1[k], dmap1[k] = l1, d1
            lmap2[k], dmap2[k] = l2, d2
        hmap1, hmap2 = preds1_ema['h_map'].clone().detach(), preds2_ema['h_map'].clone().detach()
        lmap1, dmap1 = torch.from_numpy(lmap1).cuda(), torch.from_numpy(dmap1).cuda()
        lmap2, dmap2 = torch.from_numpy(lmap2).cuda(), torch.from_numpy(dmap2).cuda()

        '''calculate loss'''
        total_loss = torch.Tensor([0]).cuda()
        targets1['hm'], targets1['dm'], targets1['lm'] = hmap1, dmap1, lmap1
        targets2['hm'], targets2['dm'], targets2['lm'] = hmap2, dmap2, lmap2
        det_total_loss1, _, _ = criterion['det'].compute_loss(preds1, targets1, targets1)
        total_loss += det_total_loss1
        det_total_loss2, losses, batch_3d_size = criterion['det'].compute_loss(preds2, targets2, targets2)
        total_loss += det_total_loss2
        targets2['batch_3d_size'] = batch_3d_size

        am_loss_hm.update(losses['det_hm'].item(), targets2['batch_size'])
        am_loss_dm.update(losses['det_dm'].item(), targets2['batch_3d_size'].item())
        am_loss_lm.update(losses['det_lm'].item(), targets2['batch_3d_size'].item())

        ''' backward and step '''
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        '''update ema model'''
        updata_ema_params(model, model_ema, 0.99, epoch * itr_num + i)
        if args.initR == 'gt':
            pass
        else:
            R12 = func.update_R(R12, R, 0.999)

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()

        if i % 10 == 0:
            log = f'Epoch {epoch} [{i}/{len(train_loader)}] lH:{am_loss_hm.avg:.5f} lD:{am_loss_dm.avg:.5f} ' \
                  f'lL:{am_loss_lm.avg:.5f} {batch_time.avg:.2f}s/batch'
            print(log)

        if DEBUG:
            if i == 1:
                break

    loss_all["lossH"].append(am_loss_hm.avg)
    loss_all["lossD"].append(am_loss_dm.avg)
    loss_all["lossL"].append(am_loss_lm.avg)


if __name__ == '__main__':
    seed_everything(1)
    main(parse_args)
