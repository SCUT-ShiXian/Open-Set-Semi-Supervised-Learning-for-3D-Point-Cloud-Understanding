from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ModelNet40_MetaDataX, ModelNet40_MetaDataU, ModelNet40_MetaDataU_StrAug, ScanObject
from model.model import PNet_Cls_Net, DGCNN_cls, DGCNN_cls_ood, DGCNN_cls_vani
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import torch.nn.functional as F

import time 

from dart_cls import Architect


def _init_():
    if not os.path.exists('log_cls'):
        os.makedirs('log_cls')
    if not os.path.exists('log_cls/'+args.log_name):
        os.makedirs('log_cls/'+args.log_name)

        os.system('cp train_cls_alphamatch.py log_cls'+'/'+args.log_name+'/'+'train_cls_alphamatch.py.backup')
        os.system('cp model/model.py log_cls' + '/' + args.log_name + '/' + 'model.py.backup')
        os.system('cp util.py log_cls' + '/' + args.log_name + '/' + 'util.py.backup')
        os.system('cp data.py log_cls' + '/' + args.log_name + '/' + 'data.py.backup')

    if not os.path.exists('log_cls/'+args.log_name+'/'+'models'):
        os.makedirs('log_cls/'+args.log_name+'/'+'models')

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def train(args, io):
    num_class = 40
                                                #### ####              meta_add_nosiy    base
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN_cls_vani(num_class=num_class).to(device)  ##### PNet_Cls_Net_param    PNet_Cls_Net   DGCNN_cls_vani  DGCNN_cls
    model.load_state_dict(torch.load('log_cls/' + args.log_name + '/models/model.t7'))

    criterion = cal_loss
    criterion_mse = nn.MSELoss(size_average=False).cuda()

    best_test_acc = 0


    ####################
    # TEST
    ####################
    with torch.no_grad():
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits, feat = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = F.cross_entropy(logits, label, reduction='mean')
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print ('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test, loss: %.6f, test acc: %.6f, val avg acc: %.6f' % (test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 1]')
    parser.add_argument('--log_name', type=str, default='gary', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--lambda_u', default=1, type=float)
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.8, type=float,
                        help='pseudo label threshold')                        
    args = parser.parse_args()

    # _init_()

    io = IOStream('log_cls/' + args.log_name + '/run.log')
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
        train(args, io)

