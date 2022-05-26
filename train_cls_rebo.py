from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ModelNet40_MetaDataX, ModelNet40_MetaDataU, ModelNet40_MetaDataU_StrAug, ModelNet40_UShapeNet_StrAug, ModelNet40_MetaDataU_GeoFea, ModelNet40_US3DIS_StrAug
from model.model import PNet_Cls_Net, DGCNN_cls_vani, DGCNN_cls, W_predictor
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import torch.nn.functional as F

import time 
import random

from torch.autograd import Variable
from dart_cls import Architect

from torch.utils.tensorboard import SummaryWriter 


def _init_():
    if not os.path.exists('log_cls'):
        os.makedirs('log_cls')
    if not os.path.exists('log_cls/'+args.log_name):
        os.makedirs('log_cls/'+args.log_name)

        os.system('cp train_cls_rebo.py log_cls'+'/'+args.log_name+'/'+'train_cls_rebo.py.backup')
        os.system('cp dart_cls.py log_cls'+'/'+args.log_name+'/'+'dart_cls.py.backup')
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

def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args, io):
    writer = SummaryWriter('log_cls/' +args.log_name + '/models')

    num_class = 40
    train_idx = random.sample(list(np.arange(100)), 50)
    val_idx = list(set(list(np.arange(100))) - set(train_idx))

    Xtrain_loader = DataLoader(ModelNet40_US3DIS_StrAug(partition='label', num_points=args.num_points, idx=train_idx), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ModelNet40_US3DIS_StrAug(partition='label', num_points=args.num_points, idx=val_idx), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    Utrain_loader = DataLoader(ModelNet40_US3DIS_StrAug(partition='unlabel', model='openset', opt=True, num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    ALLUtrain_loader = DataLoader(ModelNet40_US3DIS_StrAug(partition='unlabel', model='openset', opt=True, num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=False, drop_last=True)


    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)



    device = torch.device("cuda" if args.cuda else "cpu")
    all_unums = 9840 - 100 + 10000 + 10000

    all_weights = torch.ones(all_unums)
    fn_weights = torch.ones(all_unums).to(device)
    model = DGCNN_cls_vani(num_class=num_class).to(device)  ##### PNet_Cls_Net_param    PNet_Cls_Net   DGCNN_cls_vani  DGCNN_cls
    predictor = W_predictor(dims=2048+30).to(device)   #### 2048  1024
    predictor_teacher = W_predictor(dims=2048+30).to(device)  #### 2048  1024
    for param in predictor_teacher.parameters():
        param.detach_()


    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    prd_opt = torch.optim.Adam(predictor.parameters(), lr=args.arch_lr, weight_decay=args.arch_wd, betas=(0.5, 0.999))

    # state = torch.load('log_cls/log_name/models/state.pth')
    # model.load_state_dict(state['model'])
    # opt.load_state_dict(state['opt'])



    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    criterion_mse = nn.MSELoss(reduction='sum').cuda()

    #### dart_weight_arc 
    architect = Architect(model, predictor, predictor_teacher, prd_opt, num_class, args)
    # model = nn.DataParallel(model)


    best_test_acc = 0
    labeled_epoch = 0
    Xtrain_loader_iter = iter(Xtrain_loader)    
    for epoch in range(args.epochs):
        
        train_xloss = 0.0
        train_uloss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0

        for UWA_data, USA_data, GeoFea, U_item in (Utrain_loader):
            start_time = time.time()
            UWA_data, USA_data, GeoFea, U_item = UWA_data.to(device), USA_data.to(device), GeoFea.to(device), U_item.to(device)

            try:
                X_data, X_label = next(Xtrain_loader_iter)
            except:
                Xtrain_loader_iter = iter(Xtrain_loader)
                X_data, X_label = next(Xtrain_loader_iter)

            X_data, X_label = X_data.to(device), X_label.to(device).squeeze()

            X_data = X_data.permute(0, 2, 1)
            UWA_data = UWA_data.permute(0, 2, 1)
            USA_data = USA_data.permute(0, 2, 1)

            batch_size = X_data.size()[0]

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                logits_UWA, UWA_feat = model(UWA_data)
                sfm_label = torch.softmax(logits_UWA.detach()/args.T, dim=1)
                max_probs, pseudo_label = torch.max(sfm_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()


            if (epoch%1==0):
                predictor.train()
                predictor_teacher.train()

                data_train = Variable(USA_data, requires_grad=False)
                label_train = Variable(sfm_label, requires_grad=False)

                wl_data_val = []
                wl_lable_val = []
                for i_val in range(1):
                    try:
                        data_val, label_val = next(val_loader_iter)
                    except:
                        val_loader_iter = iter(val_loader)
                        data_val, label_val = next(val_loader_iter)
                    data_val = data_val.permute(0, 2, 1)
                    data_val = Variable(data_val, requires_grad=False).cuda()
                    label_val = Variable(label_val[:, 0], requires_grad=False).cuda(non_blocking=True)
                    wl_data_val.append(data_val)
                    wl_lable_val.append(label_val)


                w_etp = linear_rampup(epoch-30, args.epochs/3)
                unrolled_loss, loss_cos, loss_binaryU, loss_binaryX = architect.step(data_train, label_train, GeoFea, wl_data_val, wl_lable_val, U_item, scheduler.get_last_lr()[0], opt, w_etp)

            # mixup
            all_inputs = torch.cat([X_data, USA_data], dim=0)

            #### model
            opt.zero_grad()
            mix_logits, mix_feat = model(all_inputs)
            logits_X, logits_USA = mix_logits.chunk(2)
            feat_X, feat_U = mix_feat.chunk(2)

            ### weight loss
            with torch.no_grad():
                feat_U = torch.cat([feat_U, GeoFea.view(batch_size, -1)], 1)
                ### predict weights
                udata_weights = predictor(feat_U)
                fn_weights[U_item] = udata_weights.squeeze()
                weight_watch = fn_weights[U_item].reshape(batch_size, 1)

            loss_X = F.cross_entropy(logits_X, X_label, reduction='mean')

            sfm_pred = torch.softmax(logits_USA, dim=1)
            loss_U = criterion_mse(weight_watch*sfm_pred, weight_watch*sfm_label) / float(batch_size)

            w = args.lambda_u * linear_rampup(epoch, args.epochs/2.0)
            loss = loss_X + w * loss_U

            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            X_preds = logits_X.max(dim=1)[1]
            count += batch_size
            train_xloss += loss_X.item() * batch_size
            train_uloss += (loss_U).item() * batch_size

            train_true.append(X_label.cpu().numpy())
            train_pred.append(X_preds.detach().cpu().numpy())
            idx += 1
        print('fn_weights', fn_weights.shape, fn_weights.max(), fn_weights.min(), fn_weights.mean())
        print ('train total time is',total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, xloss: %.6f, uloss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                train_xloss*1.0/count,
                                                                                train_uloss*1.0/count,
                                                                                metrics.accuracy_score(
                                                                                train_true, train_pred),
                                                                                metrics.balanced_accuracy_score(
                                                                                train_true, train_pred))
        io.cprint(outstr)
        scheduler.step()

        # writer.add_scalar('loss_X', train_xloss/count, epoch)
        # writer.add_scalar('loss_U', w * train_uloss/count, epoch)


        ### reset train/val data
        train_idx = random.sample(list(np.arange(100)), 50)
        val_idx = list(set(list(np.arange(100))) - set(train_idx))

        Xtrain_loader = DataLoader(ModelNet40_US3DIS_StrAug(partition='label', num_points=args.num_points, idx=train_idx), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(ModelNet40_US3DIS_StrAug(partition='label', num_points=args.num_points, idx=val_idx), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        Xtrain_loader_iter = iter(Xtrain_loader)
        val_loader_iter = iter(val_loader)

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
                logits = F.log_softmax(logits, -1)   ##########################
                end_time = time.time()
                total_time += (end_time - start_time)
                loss = criterion(logits, label)
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
            outstr = 'Val %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                test_loss*1.0/count,
                                                                                test_acc,
                                                                                avg_per_class_acc)
            io.cprint(outstr)
        # writer.add_scalar('test_per_class_acc', avg_per_class_acc, epoch)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            print('saving ----------')
            torch.save(model.state_dict(), 'log_cls/%s/models/model.t7' % args.log_name)
            torch.save(predictor.state_dict(), 'log_cls/%s/models/predictor.t7' % args.log_name)
            state = {'model': model.state_dict(), 'opt': opt.state_dict(), 'predictor': predictor.state_dict(), 'prd_opt': prd_opt.state_dict()}
            torch.save(state, 'log_cls/%s/models/state.pth' % args.log_name)
            np.save('log_cls/' + args.log_name + '/udata_weights.npy', fn_weights.detach().cpu().numpy())

        # weights_save = {}
        # weights_save['id weights avg'] = fn_weights[0:9740].mean()
        # weights_save['id weights var'] = torch.var(fn_weights[0:9740])
        # weights_save['ood weights avg'] = fn_weights[9740:19740].mean()
        # weights_save['ood weights var'] = torch.var(fn_weights[9740:19740])
        # weights_save['noise weights avg'] = fn_weights[19740:29740].mean()
        # weights_save['noise weights var'] = torch.var(fn_weights[19740:29740])
        # writer.add_scalars('weights', weights_save, epoch)

        if epoch % 5 == 0:
            np.save('log_cls/' + args.log_name + '/udata_weights' + np.str(epoch) + '.npy', fn_weights.detach().cpu().numpy())





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
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--arch_lr', type=float, default=1e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_wd', type=float, default=3e-5, help='weight decay for arch encoding')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--prd_momentum', type=float, default=0.5, metavar='M',
                        help='predictor SGD momentum (default: 0.9)')
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

    _init_()

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

