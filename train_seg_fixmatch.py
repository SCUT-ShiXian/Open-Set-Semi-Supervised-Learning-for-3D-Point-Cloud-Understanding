from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ShapeNet, ShapeNet_PerCat, ShapeNet_Semi, ShapeNet_Semi_pi, ShapeNet_MetaData, ShapeNet_MetaDataX, ShapeNet_MetaDataU_StrAug, ShapeNet_US3DIS_StrAug
from model.model import Deform_NET, Semi_Deform_NET, Semi_PartSeg_NET, Semi_Attention_NET, DGCNN_partseg, PNet_PartSeg_Net, PNetPP_PartSeg_Net, PNet_PartSeg_Net_param
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, chamfer_loss, conss_loss, contr_loss, cal_loss_onehot, IOStream
import sklearn.metrics as metrics
import sys
sys.path.append("./emd/")
import emd_module as emd

import torch.nn.functional as F

import time 
import random
import copy

from torch.autograd import Variable
from dart_seg import Architect

def _init_():
    if not os.path.exists('log_seg'):
        os.makedirs('log_seg')
    if not os.path.exists('log_seg/'+args.log_name):
        os.makedirs('log_seg/'+args.log_name)

        os.system('cp train_seg_fixmatch.py log_seg'+'/'+args.log_name+'/'+'main.py.backup')
        os.system('cp model/model.py log_seg' + '/' + args.log_name + '/' + 'model.py.backup')
        os.system('cp util.py log_seg' + '/' + args.log_name + '/' + 'util.py.backup')
        os.system('cp data.py log_seg' + '/' + args.log_name + '/' + 'data.py.backup')

    if not os.path.exists('log_seg/'+args.log_name+'/'+'models'):
        os.makedirs('log_seg/'+args.log_name+'/'+'models')

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

label2seg = {'0':[0, 1, 2, 3],
         '1':[4, 5],
         '2':[6, 7],
         '3':[8, 9, 10, 11],
         '4':[12, 13, 14, 15],
         '5':[16, 17, 18],
         '6':[19, 20, 21],
         '7':[22, 23],
         '8':[24, 25, 26, 27],
         '9':[28, 29],
         '10':[30, 31, 32, 33, 34, 35],
         '11':[36, 37],
         '12':[38, 39, 40],
         '13':[41, 42, 43],
         '14':[44, 45, 46], 
         '15':[47, 48, 49]
         }


def calculate_shape_IoU(pred_np, seg_np, label, class_choice=None):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def train_pi_weight(args, io):
    num_classes = 16
    category_seg_num = 50 ### chair category_seg_num : 4

    Xtrain_loader = DataLoader(ShapeNet_US3DIS_StrAug(partition='label', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    Utrain_loader = DataLoader(ShapeNet_US3DIS_StrAug(partition='unlabel', model='openset', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)


    val_loader = DataLoader(ShapeNet(partition='val', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(ShapeNet(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")


    unlabel_ins_num = 12137 -100 + 10000 + 10000
    fn_weights = torch.ones(unlabel_ins_num)
    ps_model = DGCNN_partseg(part_num = category_seg_num).to(device)  ## DGCNN_partseg, PNet_PartSeg_Net, PNet_PartSeg_Net_param


    if args.use_sgd:
        print("Use SGD")
        opt_ps = optim.SGD(ps_model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt_ps = optim.Adam(ps_model.parameters(), lr=args.lr)

    scheduler_ps = CosineAnnealingLR(opt_ps, args.epochs_ps, eta_min=args.lr)

    criterion_mse = nn.MSELoss(reduction='sum').cuda()

    best_test_acc = 0
    best_test_iou = 0
    global_step = 0
    # labeled_epoch = 0
    Xtrain_loader_iter = iter(Xtrain_loader)
    for epoch in range(args.epochs_ps):
        train_loss = 0.0
        train_uloss = 0.0
        count = 0.0
        train_pred_seg = []
        train_true_seg = []
        train_label_seg = []
        total_time = 0.0

        ### training part seg net
        ps_model.train()


        for UWA_data, USA_data, U_item in (Utrain_loader):
            start_time = time.time()
            UWA_data, USA_data, U_item = UWA_data.to(device), USA_data.to(device), U_item.to(device)

            try:
                X_data, X_seg, X_label = next(Xtrain_loader_iter)
            except:
                Xtrain_loader_iter = iter(Xtrain_loader)
                X_data, X_seg, X_label = next(Xtrain_loader_iter)

            X_data, X_seg, X_label = X_data.to(device), X_seg.to(device).squeeze(), X_label.to(device).squeeze()

            #### model base
            batch_size = X_data.size()[0]
            num_pts = X_data.size()[1]

            with torch.no_grad():
                # compute guessed labels of unlabel samples

                logits_UWA, UWA_feat = ps_model(UWA_data.permute(0, 2, 1), torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda())
                logits_UWA = logits_UWA.permute(0, 2, 1).contiguous()

                sfm_seg = torch.softmax(logits_UWA.detach()/args.T, dim=2)
                max_probs, pseudo_seg = torch.max(sfm_seg, dim=-1)
                mask = max_probs.ge(args.threshold).float()


            # mixup
            all_inputs = torch.cat([X_data, USA_data], dim=0)
            all_labels = torch.cat([to_categorical(X_label, num_classes), torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda()], dim=0)
            #### model
            opt_ps.zero_grad()
            mix_logits, mix_feat = ps_model(all_inputs.permute(0, 2, 1), all_labels)
            logits_X, logits_USA = mix_logits.chunk(2)
            logits_X = logits_X.permute(0, 2, 1).contiguous()
            logits_USA = logits_USA.permute(0, 2, 1).contiguous()

            ### adding shape label
            loss_map = np.zeros((batch_size, 2048, category_seg_num), np.float32)
            for i_ins in range(batch_size):
                i_label = X_label.cpu().numpy()[i_ins]
                label_seg_idx = label2seg[str(i_label)]
                loss_map[i_ins, :, label_seg_idx] = 1.0
            logits_X[(torch.from_numpy(loss_map).to(device))==1] += 1.0


            ### weight loss
            weight_watch = fn_weights[U_item].reshape(batch_size, 1, 1).cuda()
            loss_X = F.cross_entropy(logits_X.view(-1, category_seg_num), X_seg.view(-1).squeeze(), reduction='mean')
            #### soft unsupervised loss
            sfm_pred = torch.softmax(logits_USA, dim=2)
            loss_U = criterion_mse(weight_watch*sfm_pred, weight_watch*sfm_seg) / float(batch_size * num_pts)

            w = args.lambda_u * linear_rampup(epoch, args.epochs_ps/2.0)
            loss = loss_X + w * loss_U
            loss.backward(retain_graph=True)
            opt_ps.step()
            ####
            end_time = time.time()
            total_time += (end_time - start_time)
            
            pred = logits_X.max(dim=2)[1] 
            count += batch_size
            train_loss += loss.item() * batch_size
            train_uloss += loss_U.item() * batch_size
            
            seg_np = X_seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(X_label.cpu().numpy().reshape(-1))

            global_step += 1
        scheduler_ps.step()
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, None)

        print ('train total cls time is',total_time)
        outstr = 'Train %d, loss: %.6f, train_uloss: %.6f, train iou: %.6f' % (epoch, 
                                                                                train_loss*1.0/count,
                                                                                train_uloss*1.0/count,
                                                                                np.mean(train_ious))
        io.cprint(outstr)





        ####################
        # Test (test)
        ####################
        with torch.no_grad():
            test_loss = 0.0
            count = 0.0
            ps_model.eval()
            test_true_seg = []
            test_pred_seg = []
            test_label_seg = []
            total_time = 0.0
            label = torch.from_numpy(np.zeros(args.test_batch_size, np.int64))
            for data, seg, label in test_loader:
                data, seg, label = data.to(device), seg.to(device), label.to(device).squeeze() 

                batch_size = data.size()[0]
                num_pts = data.size()[1]
                start_time = time.time()


                logits, feat = ps_model(data.permute(0, 2, 1), to_categorical(label, num_classes))
                logits = logits.permute(0, 2, 1).contiguous()
                

                loss_map = np.zeros((batch_size, 2048, category_seg_num), np.float32)
                for i_ins in range(batch_size):
                    i_label = label.cpu().numpy()[i_ins]
                    label_seg_idx = label2seg[str(i_label)]
                    loss_map[i_ins, :, label_seg_idx] = 1.0
                logits[(torch.from_numpy(loss_map).to(device))==1] += 1.0
                loss = F.cross_entropy(logits.view(-1, category_seg_num), seg.view(-1).squeeze(), reduction='mean')
                end_time = time.time()
                total_time += (end_time - start_time)


                pred = logits.max(dim=2)[1] 
                count += batch_size
                test_loss += loss.item() * batch_size


                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                test_label_seg.append(label.cpu().numpy().reshape(-1))


            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_label_seg = np.concatenate(test_label_seg)
            test_data_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, None)


            miou = np.zeros(num_classes, np.float32)
            num_miou = np.zeros(num_classes, np.float32)
            for i_ins in range(len(test_label_seg)):
                i_label = test_label_seg[i_ins]
                num_miou[i_label] += 1
                miou[i_label] += test_data_ious[i_ins]
            miou = miou / num_miou
            test_cat_ious = np.mean(miou)


            outstr = 'Val %d, loss: %.6f, data iou: %.6f, val iou: %.6f' % (epoch,
                                                                                                test_loss*1.0/count,
                                                                                                np.mean(test_data_ious),
                                                                                                test_cat_ious)
            io.cprint(outstr)


        if test_cat_ious >= best_test_iou:
            best_test_iou = test_cat_ious
            torch.save(ps_model.state_dict(), 'log_seg/%s/models/ps_model.t7' % args.log_name)




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 1]')
    parser.add_argument('--log_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='ShapeNet', metavar='N',
                        choices=['ShapeNet'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs_ps', type=int, default=250, metavar='N',
                        help='number of episode to train ')                                                
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--unrolled', type=bool, default=True, 
                        help='unrolled')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--arch_learning_rate', type=float, default=1e-3, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=3e-4, help='weight decay for arch encoding')       
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--lambda_u', default=1.0, type=float)
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.8, type=float,
                        help='pseudo label threshold')     
    args = parser.parse_args()

    _init_()

    io = IOStream('log/' + args.log_name + '/run.log')
    io.cprint(str(args))



    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train_pi_weight(args, io)
    else:
        args.model_path = 'log/' + args.log_name + '/models/model.t7'
        test(args, io)
