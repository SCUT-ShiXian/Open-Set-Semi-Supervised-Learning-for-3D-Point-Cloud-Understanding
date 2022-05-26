import torch
import torch.nn.functional as F
import numpy as np
from pointnet2_ops import pointnet2_utils
from scipy import spatial

def cal_loss(pred, gold, att_weight=None, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    if att_weight==None:
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')
    else:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb * att_weight).sum(dim=1).mean()

    return loss


def cal_loss_batch(pred, gold, att_map=None, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if att_map!=None:
        b_size = att_map.size(0)
        n_pts = att_map.size(1)
        # att_map = att_map.view(-1, n_class)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot.view(b_size, n_pts, -1)

        one_hot = torch.bmm(att_map, one_hot)
        one_hot = one_hot.view(b_size * n_pts, -1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss

def cal_loss_onehot(pred, one_hot, mask=None, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if mask!=None:
        mask = mask.view(-1, 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(mask * one_hot * log_prb).sum(dim=1).mean()
    else:
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss

def cal_percat_loss(pred, label, weight, loss_ind=0.6, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    log_prb = F.log_softmax(pred + 1e-6, dim=1)
    # print('torch.ge(label.max(dim=1)[0], 0.6, out=None)', torch.ge(label.max(dim=1)[0], 0.6, out=None).view(-1,1).shape)
    mask = torch.ge(label.max(dim=1)[0], loss_ind, out=None).view(-1,1)
    loss = -(mask * label * log_prb * weight).sum(dim=1).mean()

    return loss

def conss_loss(logit1, logit2, logit_wgt=None):
    if logit_wgt==None:
        loss_cs = ((logit1 - logit2) ** 2).sum(dim=2)
        loss_cs = loss_cs.mean()
    else:
        loss_cs = ((logit1 - logit2) ** 2).sum(dim=2)
        loss_cs = loss_cs * logit_wgt
        loss_cs = loss_cs.mean()
    return loss_cs

def pole_entropy_loss(att_map):
    ''' Calculate pole entropy loss. ''' 

    att_weight = att_map.sum(dim=2)
    att_map_nor = att_map / (torch.sum(att_map, dim=2, keepdim=True) + 1e-6)
    att_loss = (att_map_nor * F.log_softmax(att_map_nor, dim=2)).sum(dim=2)
    # print('att_weight', torch.max(att_weight), torch.min(att_weight))
    # print('att_loss', torch.max(att_loss), torch.min(att_loss))
    att_loss = att_weight * att_loss
    att_loss = att_loss.sum(dim=1).mean()

    return -att_loss

def PearsonCoeff(x,y):
    vx = x - tf.reduce_sum(x)
    vy = y - tf.reduce_sum(y)
    cost = tf.reduce_sum(vx * vy) / (tf.sqrt(tf.reduce_sum(vx ** 2)) * tf.sqrt(tf.reduce_sum(vy ** 2)))
    return cost

def corr_loss(pred, feat):

    batch_size, N, _ = pred.size()

    
    # change now


    E_same_list = []
    E_diff_list = []
    
    for c_id in range(4):
        M_c = []
        for s_id in range(batSize):
            pred_cls_s = pred_cls[s_id] #[N,4]
            mask0 = tf.boolean_mask(part_feat[s_id], tf.equal(pred_cls_s,c_id))
    #         def f1(): return part_encoder_v2(mask0) #[1,512]
            def f1(): return tf.reduce_max(mask0, axis=0) #[1,128]
            def f2(): return tf.ones([128,])
            mask0 = tf.cond(tf.reduce_any(tf.equal(pred_cls_s,c_id)), f1, f2)
            M_c.append(mask0)
            
        M_c = tf.stack(M_c) #[B,128]
        M_list.append(tf.reduce_mean(M_c, axis=0))
        corr_same_list = []
        for ii in range(batSize):
            for jj in range(ii+1, batSize):
                corr_same_list.append(PearsonCoeff(M_c[ii], M_c[jj]))
        E_same = tf.reduce_min(corr_same_list)
        E_same_list.append(E_same)
    
    for i in range(4):
        for j in range(i+1,4):
            pearson_r = PearsonCoeff(M_list[i], M_list[j])
            E_diff_list.append(pearson_r)
    
    E_corr = -tf.reduce_mean(E_same_list) + tf.reduce_max(E_diff_list)

    return corr_loss

def chamfer_loss(data1, data2):    
    ''' Calculate chamfer distance loss. '''
    # B, N, _ = data1.size()
    r1 = (data1**2).sum(dim=2, keepdim=True)
    r2 = (data2**2).sum(dim=2, keepdim=True)

    cf_dis = r1 - 2 * torch.bmm(data1, data2.permute(0, 2, 1)) + r2.permute(0, 2, 1)

    cd_loss = 0.5 * cf_dis.min(dim=1)[0] + 0.5 * cf_dis.min(dim=2)[0]
    cd_loss = cd_loss.mean()

    return cd_loss

def contr_loss(bn_feat, bn_seg):    
    ''' Calculate contrastive loss. '''
    batch_size, N, _ = bn_feat.size()

    ctr_loss = []

    for i_ins in range(batch_size):
        feat = bn_feat[i_ins]
        seg = bn_seg[i_ins]
        ctr_same_loss = []
        ctr_diff_loss = []

        unique_label = torch.unique(seg)

        for i_uni in range(len(unique_label)):
            i_label = unique_label[i_uni]
            same_idx = torch.arange(feat.shape[0])[seg==i_label]

            for j_uni in range(i_uni+1, len(unique_label)):
                j_label = unique_label[j_uni]
                diff_idx = torch.arange(feat.shape[0])[seg==j_label]

                same_feat = feat[same_idx]
                same_mean_feat = same_feat.mean(dim=0)
                diff_mean_feat = feat[diff_idx].mean(dim=0)

                same_feat_dis = ((same_feat - torch.unsqueeze(same_mean_feat, 0)) ** 2).sum(dim=1).mean()
                diff_feat_dis = ((same_mean_feat - diff_mean_feat) ** 2).sum()

                ctr_same_loss.append(same_feat_dis)
                ctr_diff_loss.append(diff_feat_dis)

                # print('same_feat_dis', same_feat_dis, 'diff_feat_dis', diff_feat_dis)

        if len(ctr_same_loss)>0:
            ctr_same_loss = torch.tensor(ctr_same_loss)
            if len(ctr_diff_loss) > 0:
                ctr_diff_loss = torch.tensor(ctr_diff_loss)
                ctr_diff_loss = ctr_diff_loss / (ctr_diff_loss.max() + 1e-6) - 1.0
                ctr_loss.append(ctr_same_loss.mean() - ctr_diff_loss.mean())
            else:
                ctr_loss.append(ctr_same_loss.mean())
    return torch.tensor(ctr_loss).mean()

def contr_loss2(bn_feat, bn_seg):    
    ''' Calculate contrastive loss. '''
    batch_size, N, _ = bn_feat.size()

    # print('bn_feat', bn_feat.shape)
    # print('bn_seg', bn_seg.shape)

    ctr_same_loss = []
    ctr_diff_loss = []

    for i_ins in range(batch_size):
        feat = bn_feat[i_ins]
        seg = bn_seg[i_ins]

        for i_label in torch.unique(seg):
            same_idx = torch.arange(feat.shape[0])[seg==i_label]
            diff_idx = torch.arange(feat.shape[0])[seg!=i_label]


            same_feat = feat[same_idx]
            same_feat_dis = ((torch.unsqueeze(same_feat, 1) - torch.unsqueeze(feat, 0)) ** 2).sum(dim=2) 
            same_feat_dis = same_feat_dis / same_feat_dis.max()
            # diff_feat = feat[diff_idx]
            # diff_feat_dis = ((torch.unsqueeze(diff_feat, 1) - torch.unsqueeze(feat, 0)) ** 2).sum(dim=2)

            ctr_same_loss.append((same_feat_dis[:, same_idx].min(dim=1)[0]).mean())
            ctr_diff_loss.append((same_feat_dis[:, diff_idx].min(dim=1)[0]).min() - 1.0)

    ctr_loss = ctr_same_loss.mean() - ctr_diff_loss.mean()
    return ctr_loss

def deform_abs_loss(df_xyz, att_map):
    ''' calculate |deform|**2 '''
    att_map_weight = att_map.sum(dim=2)

    df_abs_loss = (df_xyz**2).sum(dim=2)
    df_abs_loss = (att_map_weight * df_abs_loss).mean(dim=1) 
    df_abs_loss = df_abs_loss.mean()
    return df_abs_loss

def abs_loss(logits1, logits2, att_map1, att_map2):

    att_map_norm1 = att_map1 / (torch.sum(att_map1, dim=2, keepdim=True) + 1e-6)
    att_map_norm2 = att_map2 / (torch.sum(att_map2, dim=2, keepdim=True) + 1e-6)

    cp_lg1 = torch.bmm(att_map_norm1, logits2)
    cp_lg2 = torch.bmm(att_map_norm1, logits1)

    loss1 = ((cp_lg1-logits1)**2).sum(dim=2) 
    loss2 = ((cp_lg2-logits2)**2).sum(dim=2) 

    loss = (loss1 + loss2).mean(dim=1)
    loss = loss.mean()
    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    xyz = xyz.contiguous()

    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points