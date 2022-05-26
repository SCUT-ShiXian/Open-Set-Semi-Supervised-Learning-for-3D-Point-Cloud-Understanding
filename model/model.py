import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from model.pointnet import STN3d, STNkd, feature_transform_reguliarzer, PointNetEncoder

from util import sample_and_group 
from torch.autograd import Variable

import math

class Deform_NET(nn.Module):
    def __init__(self):
        super(Deform_NET, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(3, 16, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))                                
        self.conv4 = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.conv1_tp = nn.Sequential(nn.Conv1d(3, 16, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2_tp = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3_tp = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))                                
        self.conv4_tp = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5_tp = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))



        self.conv1_mix = nn.Sequential(nn.Conv1d(2051, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2_mix = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.conv_df = nn.Conv1d(128, 3, kernel_size=1, bias=False)

        # self.partseg = PartSeg_NET(4)

    def forward(self, x, tp_x):
        batch_size, N, _ = x.size()
        xyz = x.permute(0, 2, 1)
        tp_xyz = tp_x.permute(0, 2, 1)  ### b 3 n


        x = self.conv1(xyz)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)

        tp_x = self.conv1_tp(tp_xyz)
        tp_x = self.conv2_tp(tp_x)
        tp_x = self.conv3_tp(tp_x)
        tp_x = self.conv4_tp(tp_x)
        tp_x = self.conv5_tp(tp_x)
        tp_x = F.adaptive_max_pool1d(tp_x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)



        mix = torch.cat([tp_xyz, tp_x, x], dim=1)
        mix = self.conv1_mix(mix)
        mix = self.conv2_mix(mix)
        df_v = self.conv_df(mix)

        df_x = df_v + tp_xyz

        # pred, feat = self.partseg(df_x)
        # return pred, feat, df_v
        return df_x, df_v

class PartSeg_NET(nn.Module):
    def __init__(self, part_num=4):
        super(PartSeg_NET, self).__init__()
        self.part_num = part_num

        self.conv1 = nn.Sequential(nn.Conv1d(3, 16, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))                                
        self.conv4 = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.conv6 = nn.Sequential(nn.Conv1d(1027, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(259, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(131, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv9 = nn.Conv1d(128, 4, kernel_size=1, bias=False)



    def forward(self, df_x):

        batch_size, _, N = df_x.size()

        x = self.conv1(df_x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        max_x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        mix = torch.cat([df_x, max_x], dim=1)
        mix = self.conv6(mix)
        mix = torch.cat([df_x, mix], dim=1)
        mix = self.conv7(mix)
        mix = torch.cat([df_x, mix], dim=1)
        feat = self.conv8(mix)
        pred = self.conv9(feat)    

        return pred, feat

class Semi_Deform_NET(nn.Module):
    def __init__(self):
        super(Semi_Deform_NET, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(3, 16, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))                                
        self.conv4 = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.conv1_tp = nn.Sequential(nn.Conv1d(3, 16, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2_tp = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3_tp = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))                                
        self.conv4_tp = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5_tp = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))



        self.conv1_mix = nn.Sequential(nn.Conv1d(2051, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2_mix = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.conv_df = nn.Conv1d(128, 3, kernel_size=1, bias=False)

        self.partseg = Semi_PartSeg_NET(50)

        self.attmap1 = AMAP_Layer(128)
        self.attmap2 = AMAP_Layer(128)

    def forward(self, x, tp_x, cls_label):
        batch_size, N, _ = x.size()
        xyz = x.permute(0, 2, 1)
        tp_xyz = tp_x.permute(0, 2, 1)  ### b 3 n


        x = self.conv1(xyz)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)

        tp_x = self.conv1_tp(tp_xyz)
        tp_x = self.conv2_tp(tp_x)
        tp_x = self.conv3_tp(tp_x)
        tp_x = self.conv4_tp(tp_x)
        tp_x = self.conv5_tp(tp_x)
        tp_x = F.adaptive_max_pool1d(tp_x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)



        mix = torch.cat([tp_xyz, tp_x, x], dim=1)
        # print('mix',mix.shape)
        mix = self.conv1_mix(mix)
        mix = self.conv2_mix(mix)
        # print('mix',mix.shape)
        df_v = self.conv_df(mix)
        df_x = df_v + tp_xyz

        pred, feat = self.partseg(df_x, cls_label)
        pred_x, feat_x = self.partseg(xyz, cls_label)
        pred_tp_x, feat_tp_x = self.partseg(tp_xyz, cls_label)

        feat1, feat2, attention1, attention2 = self.attmap1(feat_tp_x, feat)
        feat1, feat2, attention1, attention2 = self.attmap2(feat1, feat2)
        feat3, feat4, attention3, attention4 = self.attmap1(feat, feat_x)
        feat3, feat4, attention3, attention4 = self.attmap2(feat3, feat4)

        return pred, pred_tp_x, pred_x, feat, feat_tp_x, feat_x, attention1, attention2, attention3, attention4, df_v
        # return pred, pred_tp_x, feat, feat_tp_x, attention1, attention2, attention3, attention4, df_v
        # return pred, pred_ori, feat, feat_ori, df_v

class Semi_PartSeg_NET(nn.Module):
    def __init__(self, part_num=50):
        super(Semi_PartSeg_NET, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(3, 16, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))                                
        self.conv4 = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv6 = nn.Sequential(nn.Conv1d(1024 + 3 + 64, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(259, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(131, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv9 = nn.Conv1d(128, part_num, kernel_size=1, bias=False)



    def forward(self, df_x, cls_label):

        batch_size, _, N = df_x.size()

        x = self.conv1(df_x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        max_x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # avg_x = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(batch_size,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        mix = torch.cat([df_x, max_x, cls_label_feature], dim=1)
        mix = self.conv6(mix)
        mix = torch.cat([df_x, mix], dim=1)
        mix = self.conv7(mix)
        mix = torch.cat([df_x, mix], dim=1)
        feat = self.conv8(mix)
        pred = self.conv9(feat)    

        return pred, feat

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class DGCNN_partseg(nn.Module):
    def __init__(self, part_num = 50):
        super(DGCNN_partseg, self).__init__()
        self.part_num = part_num
        self.k = 40
        self.transform_net = Transform_Net()
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.part_num, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        feat = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(feat)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x, torch.max(feat, dim=-1)[0]

    def new(self):
        model_new = DGCNN_partseg(part_num=self.part_num).cuda()
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new




class DGCNN_partseg_param(nn.Module):
    def __init__(self, part_num = 50, ins_num=1080):
        super(DGCNN_partseg_param, self).__init__()
        self.part_num = part_num
        self.k = 40
        self.ins_num = ins_num
        self.transform_net = Transform_Net()
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(8, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.part_num, kernel_size=1, bias=False)

        self._initialize_alphas(self.ins_num)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        feat = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(feat)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x, feat

    def _initialize_alphas(self, ins_num):
        alphas = torch.ones((ins_num), dtype=torch.float32)
        self.alphas_normal = Variable(alphas.cuda(), requires_grad=True)
        self._arch_parameters = [
        self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def new(self):
        model_new = DGCNN_partseg_param(part_num=self.part_num, ins_num=self.ins_num).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

class Wseg_predictor(nn.Module):
    def __init__(self, dims):
        super(Wseg_predictor, self).__init__()
        self.bnf = nn.BatchNorm1d(256)
        self.convf = nn.Sequential(nn.Conv1d(dims, 256, kernel_size=1, bias=False),
                                   self.bnf,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dpf = nn.Dropout(p=0.5)


        self.linear1 = nn.Linear(512, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 1)


    def forward(self, feat):
        batch_size = feat.size(0)
        feat = self.convf(feat)                       # (batch_size, 1280, num_points) -> (batch_size, 256, num_points)
        feat = self.dpf(feat)

        x1 = F.adaptive_max_pool1d(feat, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(feat, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feat = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        uscores = F.leaky_relu(self.bn1(self.linear1(feat)), negative_slope=0.2)
        uscores = self.dp1(uscores)
        uscores = F.leaky_relu(self.bn2(self.linear2(uscores)), negative_slope=0.2)
        uscores = self.dp2(uscores)
        uscores = self.linear3(uscores)  
        return uscores

class Semi_Attention_NET(nn.Module):
    def __init__(self):
        super(Semi_Attention_NET, self).__init__()

        self.attmap1 = AMAP_Layer(128)
        self.attmap2 = AMAP_Layer(128)
        self.attmap3 = AMAP_Layer(128)
        self.attmap4 = AMAP_Layer(128)

    def forward(self, x_feat, df_feat):

        feat, featb, att1, att1b = self.attmap1(x_feat, df_feat)
        feat, featb, att2, att2b = self.attmap2(feat, featb)
        feat, featb, att3, att3b = self.attmap3(feat, featb)
        feat, featb, att4, att4b = self.attmap4(feat, featb)

        atta = att1 + att2 + att3 + att4
        attb = att1b + att2b + att3b + att4b

        return feat, featb, atta, attb

class Semi_Cls_NET(nn.Module):
    def __init__(self, output_channels=40):
        super(Semi_Cls_NET, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        feat = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(feat)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x, feat

class PNet_PartSeg_Net_param(nn.Module):
    def __init__(self, part_num=50, ins_num=1080, normal_channel=False):
        super(PNet_PartSeg_Net_param, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.ins_num = ins_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4936, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)

        self._initialize_alphas(self.ins_num)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+8, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        # net = net.transpose(2, 1).contiguous()
        # net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        # net = net.view(B, N, self.part_num) # [B, N, 50]
        
        return net, trans_feat

    def _initialize_alphas(self, ins_num):
        alphas = torch.ones((ins_num), dtype=torch.float32)
        self.alphas_normal = Variable(alphas.cuda(), requires_grad=True)
        self._arch_parameters = [
        self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def new(self):
        model_new = PNet_PartSeg_Net_param(part_num=self.part_num, ins_num=self.ins_num).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

class PNet_PartSeg_Net(nn.Module):
    def __init__(self, part_num=50, normal_channel=False):
        super(PNet_PartSeg_Net, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4936, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        # feat = torch.cat([out1, out2, out3], 1)

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+8, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        feat = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(feat)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        # net = net.transpose(2, 1).contiguous()
        # net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        # net = net.view(B, N, self.part_num) # [B, N, 50]
        
        return net, feat

    def new(self):
        model_new = PNet_PartSeg_Net(part_num=self.part_num).cuda()
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new

class PNetPP_PartSeg_Net(nn.Module):
    def __init__(self, num_classes=50, normal_channel=False):
        super(PNetPP_PartSeg_Net, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x, l3_points

class PNet_Cls_Net(nn.Module):
    def __init__(self, num_class=40, normal_channel=False):
        super(PNet_Cls_Net, self).__init__()
        self.num_class = num_class
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        feat, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x, feat

    def new(self):
        model_new = PNet_Cls_Net(num_class=self.num_class).cuda()
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new

class PNet_Cls_Net_param(nn.Module):
    def __init__(self, num_class=40, ins_num=2400, normal_channel=False):
        super(PNet_Cls_Net_param, self).__init__()
        self.ins_num = ins_num
        self.num_class = num_class
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self._initialize_alphas(self.ins_num)

    def forward(self, x):
        feat, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x, feat

    def _initialize_alphas(self, ins_num):
        alphas = torch.ones((ins_num), dtype=torch.float32)
        self.alphas_normal = Variable(alphas.cuda(), requires_grad=True)
        self._arch_parameters = [
        self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def new(self):
        model_new = PNet_Cls_Net_param(num_class=self.num_class, ins_num=self.ins_num).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

class PNet_Cls_Net_ood1(nn.Module):
    def __init__(self, num_class=40, normal_channel=False):
        super(PNet_Cls_Net_ood1, self).__init__()
        self.num_class = num_class
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.fc4 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        feat, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x1 = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x1, feat

class PNet_Cls_Net_ood(nn.Module):
    def __init__(self, num_class=40, normal_channel=False):
        super(PNet_Cls_Net_ood, self).__init__()
        self.num_class = num_class
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.fc4 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        feat, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x1 = self.fc3(x)
        x2 = self.fc4(x)
        # x = F.log_softmax(x, dim=1)
        return x1, x2, feat

class PNetPP_Cls_Net(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(PNetPP_Cls_Net, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points

class ULW_prd_Net(nn.Module):
    def __init__(self, num_class=40, dims=256):
        super(ULW_prd_Net, self).__init__()
        self.num_class = num_class
        self.start = torch.zeros(num_class)
        self.mbank = torch.Tensor(num_class, dims).cuda()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, dims)
        self.fc3 = nn.Linear(num_class + dim, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)
        # self.bn0 = nn.BatchNorm1d(dims)

        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, lufeat, llabel):
        batch_size = lufeat.size(0) // 2

        lufeat = self.fc1(lufeat)
        lufeat = self.fc2(lufeat)

        lufeat = F.normalize(lufeat, p=2, dim=1)
        # lufeat = self.bn0(lufeat)
        lfeat = lufeat[:batch_size]
        ufeat = lufeat[batch_size:]

        for i in range(batch_size):
            i_l = llabel[i]
            if self.start[i_l] == 0:
                self.mbank[i_l] = lfeat[i].detach()
                self.start[i_l] = 1
            else:
                self.mbank[i_l]= 0.9*self.mbank[i_l] + 0.1*lfeat[i].detach()


        lm_matrix = torch.matmul(lfeat, torch.transpose(self.mbank,0,1))  #### bd * dm = bm
        lm_matrix = torch.norm(lfeat.reshape(batch_size, 1, -1) - self.mbank.reshape(1, self.num_class, -1), dim=-1, keepdim=False)  #### norm2L(b1d - 1md) = bm
        lscores = torch.min(lm_matrix, dim=-1)[0]  ## bm = b

        um_matrix = torch.matmul(ufeat, torch.transpose(self.mbank,0,1))  #### bd * dm = bm
        um_matrix = torch.norm(ufeat.reshape(batch_size, 1, -1) - self.mbank.reshape(1, self.num_class, -1), dim=-1, keepdim=False)  #### norm2L(b1d - 1md) = bm
        # um_matrix = torch.min(um_matrix, dim=-1)[0]  ## bm = b

        ufeat = torch.concat([ufeat, um_matrix], dim=-1)  #### bd + bm = b(d+m)
        uscores = F.relu(self.bn1(self.fc3(ufeat)))
        uscores = F.relu(self.bn2(self.dropout(self.fc4(uscores))))
        uscores = self.fc5(uscores)

        return lscores, uscores, self.mbank

class DGCNN_cls(nn.Module):
    def __init__(self, num_class=40, ins_num=2400):
        """
        dgcnn classification network
        """
        super(DGCNN_cls, self).__init__()
        self.k = 20
        self.num_class = num_class
        self.ins_num = ins_num

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)

        self._initialize_alphas(self.ins_num)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feature = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(feature)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, num_class)
        
        return x, feature

    def _initialize_alphas(self, ins_num):
        alphas = torch.ones((ins_num), dtype=torch.float32)
        self.alphas_normal = Variable(alphas.cuda(), requires_grad=True)
        self._arch_parameters = [
        self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def new(self):
        model_new = DGCNN_cls(num_class=self.num_class, ins_num=self.ins_num).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

class DGCNN_cls_vani(nn.Module):
    def __init__(self, num_class=40):
        """
        dgcnn classification network
        """
        super(DGCNN_cls_vani, self).__init__()
        self.k = 20
        self.num_class = num_class

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feature = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(feature)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, num_class)
        
        return x, feature

    def new(self):
        model_new = DGCNN_cls_vani(num_class=self.num_class).cuda()
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new



class DGCNN_cls_OPMATCH(nn.Module):
    def __init__(self, num_class=40):
        """
        dgcnn classification network
        """
        super(DGCNN_cls_OPMATCH, self).__init__()
        self.k = 20
        self.num_class = num_class

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        self.linear4 = nn.Linear(256, 2*num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feature = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(feature)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x1 = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, num_class)
        x2 = self.linear4(x)                                             # (batch_size, 256) -> (batch_size, num_class)
        return x1, x2, feature

    def new(self):
        model_new = DGCNN_cls_vani(num_class=self.num_class).cuda()
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new

class DGCNN_cls_ood1(nn.Module):
    def __init__(self, num_class=40, ins_num=2400):
        """
        dgcnn classification network
        """
        super(DGCNN_cls_ood1, self).__init__()
        self.k = 20
        self.num_class = num_class
        self.ins_num = ins_num

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feature = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(feature)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x1 = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, num_class)
        return x1, feature

class DGCNN_cls_ood(nn.Module):
    def __init__(self, num_class=40, ins_num=2400):
        """
        dgcnn classification network
        """
        super(DGCNN_cls_ood, self).__init__()
        self.k = 20
        self.num_class = num_class
        self.ins_num = ins_num

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feature = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(feature)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x1 = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, num_class)
        x2 = self.linear4(x)   
        return x1, x2, feature

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                # print('name_t', name_t, 'param_t', param_t.shape, 'grad', grad.shape)
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)





    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    # print('tttt')
                    self.set_param(mod, rest, param)
                    break
        else:
            # print('curr_mod', curr_mod)
            # print('name', name)
            # print('param', param.shape)
            # param = torch.nn.parameter.Parameter(param, requires_grad=True)
            # print('param', param, param.shape)
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        self.update_batch_stats = True

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        #if self.update_batch_stats:
        #    return super().forward(x)
        #else:
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class DS3L_DGCNN_cls(MetaModule):
    def __init__(self, num_class=40):
        """
        dgcnn classification network
        """
        super(DS3L_DGCNN_cls, self).__init__()
        self.k = 20
        self.num_class = num_class

        self.bn1 = MetaBatchNorm2d(64)
        self.bn2 = MetaBatchNorm2d(64)
        self.bn3 = MetaBatchNorm2d(128)
        self.bn4 = MetaBatchNorm2d(256)
        self.bn5 = MetaBatchNorm2d(1024)

        self.conv1 = MetaConv2d(3*2, 64, kernel_size=1, bias=False)
        self.conv2 = MetaConv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv3 = MetaConv2d(64*2, 128, kernel_size=1, bias=False)
        self.conv4 = MetaConv2d(128*2, 256, kernel_size=1, bias=False)
        self.conv5 = MetaConv2d(512, 1024, kernel_size=1, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)
        # self.conv1 = nn.Sequential(MetaConv2d(6, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(MetaConv2d(64*2, 64, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(MetaConv2d(64*2, 128, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv4 = nn.Sequential(MetaConv2d(128*2, 256, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv5 = nn.Sequential(MetaConv2d(512, 1024, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = MetaLinear(1024*2, 512)
        self.bn6 = MetaBatchNorm2d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = MetaLinear(512, 256)
        self.bn7 = MetaBatchNorm2d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = MetaLinear(256, num_class)


        # self.conv1 = MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.dense = torch.nn.Sequential(nn.Dropout(p=0.5),
        #                                 MetaLinear(32 * 7 * 7, n_out))

    def forward(self, x):
        batch_size = x.size(0)


        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # x = torch.unsqueeze(x, -1)  
        x = self.relu1(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # x = torch.unsqueeze(x1, -1)  
        x = self.relu2(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # x = torch.unsqueeze(x2, -1)  
        x = self.relu3(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        # x = torch.unsqueeze(x3, -1)  
        x = self.relu4(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)
        x = torch.unsqueeze(x, -1)              # (batch_size, 64+64+128+256, num_points) ->  (batch_size, 64+64+128+256, num_points, 1)

        x = self.relu5(self.bn5(self.conv5(x)))
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, emb_dims, num_points, 1) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feature = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)


        x = F.leaky_relu(self.bn6(self.linear1(feature)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, num_class)
        
        return x

class CNN(MetaModule):
    def __init__(self, n_out=40):
        super(CNN, self).__init__()

        # self.conv = torch.nn.Sequential(MetaConv2d(1, 16, 3, padding=1),
        #                                 nn.MaxPool2d(3, stride=2, padding=1),
        #                                 nn.ReLU(),
        #                                 MetaConv2d(16, 32, 3, padding=1),
        #                                 nn.MaxPool2d(3, stride=2, padding=1),
        #                                 nn.ReLU()
        #                                 )
        # self.dense = torch.nn.Sequential(nn.Dropout(p=0.5),
        #                                 MetaLinear(32 * 7 * 7, n_out))
        self.dense = torch.nn.Sequential(nn.Dropout(p=0.5),
                                        MetaLinear(3, n_out))
    def forward(self, x):
        # x = self.conv(x)
        # x = x.view(-1, 32 * 7 * 7)
        print('x', x.shape)
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.dense(x)
        return x


class WNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(WNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        # self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)




class W_predictor(nn.Module):
    def __init__(self, dims):
        super(W_predictor, self).__init__()

        self.linear1 = nn.Linear(dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)

        self.linear3 = nn.Linear(256, 1)
        # self.linear4 = nn.Linear(1, 1, bias=False)
    def forward(self, feat):
        uscores = F.leaky_relu(self.bn1(self.linear1(feat)), negative_slope=0.2)
        # uscores = self.dp1(uscores)
        uscores = F.leaky_relu(self.bn2(self.linear2(uscores)), negative_slope=0.2)
        # uscores = self.dp2(uscores)
        uscores = self.linear3(uscores)
        uscores = torch.sigmoid(uscores)       
        # weights = self.linear4(uscores) 
        return uscores#, weights

class W_scale(nn.Module):
    def __init__(self):
        super(W_scale, self).__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
    def forward(self):
        return self.scale

class Pct_Cls_NET(nn.Module):
    def __init__(self, output_channels=40):
        super(Pct_Cls_NET, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        global_x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(global_x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x, global_x

class Pct_PartSeg_Net(nn.Module):
    def __init__(self, part_num=50):
        super(Pct_PartSeg_Net, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 50, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)


        x_max = F.adaptive_max_pool1d(x, 1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x_avg = F.adaptive_avg_pool1d(x, 1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(batch_size,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat([x_max_feature, x_avg_feature, cls_label_feature], dim=1) # 1024 + 64
        x = torch.cat([x, x_global_feature], dim=1) # 1024 * 3 + 64 
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # print('attention', attention.shape)
        
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class AMAP_Layer(nn.Module):
    def __init__(self, channels):
        super(AMAP_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, tp_x):
        # x: b, c, n
        x_q = self.q_conv(x).permute(0, 2, 1)
        tp_x_q = self.q_conv(tp_x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        tp_x_k = self.k_conv(tp_x)
        # b, n, n
        energy1 = torch.bmm(x_q, tp_x_k)
        energy2 = torch.bmm(tp_x_q, x_k)

        attention1 = self.softmax(energy1)
        attention1 = attention1 / (1e-9 + attention1.sum(dim=1, keepdim=True))
        attention2 = self.softmax(energy2)
        attention2 = attention2 / (1e-9 + attention2.sum(dim=1, keepdim=True))


        # b, c, n
        x_r = torch.bmm(self.v_conv(tp_x), attention1)
        x_r = self.act(self.after_norm(self.trans_conv(tp_x - x_r)))
        x = x + x_r
        # b, c, n
        tp_x_r = torch.bmm(self.v_conv(x), attention2)
        tp_x_r = self.act(self.after_norm(self.trans_conv(x - tp_x_r)))
        tp_x = tp_x + tp_x_r

        return x, tp_x, attention1, attention2


