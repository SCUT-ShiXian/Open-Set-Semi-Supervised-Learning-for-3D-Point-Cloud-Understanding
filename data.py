import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

import random
import torch
random.seed(1)
import time


label2seg = {'0':[0, 1, 2, 3],   # Airplane
         '1':[4, 5],  # Bag
         '2':[6, 7],  # Cap
         '3':[8, 9, 10, 11],  # Car
         '4':[12, 13, 14, 15],  # Chair
         '5':[16, 17, 18],  # Earphone
         '6':[19, 20, 21],  # Guitar
         '7':[22, 23],  # Knife
         '8':[24, 25, 26, 27],  # Lamp
         '9':[28, 29],  # Laptop
         '10':[30, 31, 32, 33, 34, 35],  # Motorbike
         '11':[36, 37],  # Mug
         '12':[38, 39, 40],  # Pistol
         '13':[41, 42, 43],  # Rocket
         '14':[44, 45, 46],   # Skateboard
         '15':[47, 48, 49]  # Table
         }

MS_overlap_label = [0, 3, 4, 6, 8, 9, 15]
MO_overlap_label = [1,5,6,8,11,12,13,14]
OS_overlap_label = [1,4,15]
# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], \
#     'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], \
#         'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}


def download(DATA_DIR):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def normal_data(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def load_data(partition):
    DATA_DIR = '/data3/data4_lab-shi.xian/dataset'
    download(DATA_DIR)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    print('all_data', all_data.shape)
    print('all_label', all_label.shape)


    # unique_label = np.unique(all_label)
    # print('unique_label', unique_label)
    # trainl_data = []
    # trainl_label = []
    # trainu_data = []
    # trainu_label = []
    # val_data = []
    # val_label = []
    # test_data = []
    # test_label = []

    # idx_pick = [0, 2, 3, 4, 7, 9, 12, 14, 16, 17, 21, 22, 23, 25, 26, 30, 33, 34, 35, 37]
    # print('idx_pick', len(idx_pick))
    # idx_all = np.arange(all_data.shape[0])

    # for i_label in idx_pick:
    #     one_label_idx = idx_all[all_label[:,0]==i_label]
    #     sample_idx = np.array(random.sample(list(one_label_idx), 140))
    #     trainl_data.append(all_data[sample_idx[:20]])
    #     trainl_label.append(all_label[sample_idx[:20]])
    #     trainu_data.append(all_data[sample_idx[20:40]])
    #     trainu_label.append(all_label[sample_idx[20:40]])
    #     val_data.append(all_data[sample_idx[40:90]])
    #     val_label.append(all_label[sample_idx[40:90]])
    #     test_data.append(all_data[sample_idx[90:140]])
    #     test_label.append(all_label[sample_idx[90:140]])

    # trainl_data = np.concatenate(trainl_data)
    # trainl_label = np.concatenate(trainl_label)
    # trainu_data = np.concatenate(trainu_data)
    # trainu_label = np.concatenate(trainu_label)
    # val_data = np.concatenate(val_data)
    # val_label = np.concatenate(val_label)
    # test_data = np.concatenate(test_data)
    # test_label = np.concatenate(test_label)
    # print('trainl_data', trainl_data.shape)
    # print('trainl_label', trainl_label.shape)
    # print('trainl unique', np.unique(trainl_label))
    # print('trainu_data', trainu_data.shape)
    # print('trainu_label', trainu_label.shape)
    # print('trainu unique', np.unique(trainu_label))
    # print('val_data', val_data.shape)
    # print('val_label', val_label.shape)
    # print('val unique', np.unique(val_label))
    # print('test_data', test_data.shape)
    # print('test_label', test_label.shape)
    # print('test unique', np.unique(test_label))
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_ltrain_data.npy', trainl_data)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_ltrain_label.npy', trainl_label)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_utrain_data.npy', trainu_data)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_utrain_label.npy', trainu_label)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_val_data.npy', val_data)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_val_label.npy', val_label)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_test_data.npy', test_data)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/meta_test_label.npy', test_label)


    # all_add_idx = []
    # for i_label in range(40):
    #     if i_label not in idx_pick:
    #         one_label_idx = idx_all[all_label[:,0]==i_label]
    #         all_add_idx.append(one_label_idx)
    # all_add_idx = np.concatenate(all_add_idx)
    # print('all_add_idx', all_add_idx.shape)
    # sample_idx = np.array(random.sample(list(all_add_idx), 1000))
    # add_data = all_data[sample_idx]
    # add_label = all_label[sample_idx]
    # print('add_data', add_data.shape)
    # print('add unique', np.unique(add_label))
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/add_data.npy', add_data)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/add_label.npy', add_label)

    # noisy_data = add_data.copy()
    # for i_ins in range(noisy_data.shape[0]):
    #     # add_noisy_utrain_data[i_ins] = pitran_pointcloud(add_noisy_utrain_data[i_ins], jt1=0.05, jt2=0.1)
    #     noisy_data[i_ins] = normal_data(np.clip(np.random.randn(2048, 3), -1.0, 1.0))

    # print('noisy_data', noisy_data.shape)
    # np.save('data_cls/meta_data_l20u20v50test50add1000noisy1000/noisy_data.npy', noisy_data)
    
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875, min_dropout_ratio=0.0):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    dropout_ratio = max(dropout_ratio, min_dropout_ratio)
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud, l1=2./3., l2=-0.2, h1=3./2., h2=0.2):
    xyz1 = np.random.uniform(low=l1, high=h1, size=[3])
    xyz2 = np.random.uniform(low=l2, high=h2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud, sigma=0.01):
    angle = (2*np.random.rand() - 1) * np.pi * sigma
    direct_idx = random.sample([0,1,2], 1)

    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if direct_idx==0:
        pointcloud.dot(R_x).astype('float32')
    elif direct_idx==1:
        pointcloud.dot(R_y).astype('float32')
    elif direct_idx==2:
        pointcloud.dot(R_z).astype('float32')
    return pointcloud

def random_mirror_pointcloud(pointcloud):
    mirror_opt = np.random.choice([0,1])
    if mirror_opt == 0:
        pass
    elif mirror_opt == 1:
        pointcloud[:,2] = -pointcloud[:,2]
    return pointcloud

def pitran_pointcloud(pointcloud, jt1=0.01, jt2=0.02):
    pointcloud = jitter_pointcloud(pointcloud, sigma=jt1, clip=jt2)
    pointcloud = translate_pointcloud(pointcloud)
    pointcloud = rotate_pointcloud(pointcloud, sigma=0.01)
    pointcloud = random_mirror_pointcloud(pointcloud)
    return pointcloud

def cutout_pointcloud(pointcloud, L=0.5):
    L = L * np.random.rand()
    ord_x = 2.0 * np.random.rand() - 1.0
    ord_y = 2.0 * np.random.rand() - 1.0
    ord_z = 2.0 * np.random.rand() - 1.0
    ord_x2 = ord_x + L
    ord_y2 = ord_y + L
    ord_z2 = ord_z + L

    cut_sign = np.ones(pointcloud.shape[0], int)
    cut_sign = cut_sign * [pointcloud[:,0]>ord_x] * [pointcloud[:,0]<ord_x2] * [pointcloud[:,1]>ord_y] * [pointcloud[:,1]<ord_y2] * [pointcloud[:,2]>ord_z] * [pointcloud[:,2]<ord_z2]
    
    patch = np.clip(0.5 * np.random.randn(np.sum(cut_sign), 3), -1.0, 1.0)
    pointcloud[cut_sign[0,:]==1] = patch
    return pointcloud

def cut_one_axis(pc):
    xy_opt = np.random.choice([0,1,2,3,4,5])  #### xyz 012
    cut_range = np.clip(np.random.rand(), 0.2, 0.8)

    if (xy_opt == 0):
        pc[:, 0] = np.clip(pc[:, 0], -cut_range, 1.0)
    elif (xy_opt == 1):
        pc[:, 0] = np.clip(pc[:, 0], -1.0, cut_range)
    elif (xy_opt == 2):
        pc[:, 1] = np.clip(pc[:, 1], -cut_range, 1.0)
    elif (xy_opt == 3):
        pc[:, 1] = np.clip(pc[:, 1], -1.0, cut_range)
    elif (xy_opt == 4):
        pc[:, 2] = np.clip(pc[:, 2], -cut_range, 1.0)
    elif (xy_opt == 5):
        pc[:, 2] = np.clip(pc[:, 2], -1.0, cut_range)
    return pc

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:]
  label = f['label'][:]
  return (data, label)

def loadDataFile(filename):
  return load_h5(filename)

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    num_data = data.shape[0]
    data_idx = np.arange(0,num_data)
    return (data, label, seg, num_data, data_idx)

def compute_eign(pc):
    covariance = np.cov(pc.T)
    eignvalue, eignvector = np.linalg.eig(covariance)
    eignvalue = np.real(eignvalue)
    return eignvalue, eignvector


def load_S3DIS_blocks():

    folders = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
    n_labels = 13


    Files_base_path = '/data3/data4_lab-shi.xian/dataset/'
    ALL_FILES = getDataFiles(Files_base_path + 'indoor3d_sem_seg_hdf5_data/all_files.txt') 
    room_filelist = [line.rstrip() for line in open(Files_base_path + 'indoor3d_sem_seg_hdf5_data/room_filelist.txt')] 
    print(len(room_filelist))


    # Load ALL data
    room_name_list = []
    for num_pc, room_name in enumerate(room_filelist):
        if room_name not in room_name_list:
            room_name_list.append(room_name)
    # print('room_name_list', room_name_list, 'num_rooms', len(room_name_list))


    data_batch_list = []
    label_batch_list = []
    for h5_filename in ALL_FILES:
        # print('h5_filename:', h5_filename)
        data_batch, label_batch = loadDataFile(Files_base_path + h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    return data_batches[:, :2048, :3], label_batches



class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ModelNet40_US3DIS_StrAug(Dataset):
    def __init__(self, num_points, idx=[], partition='label', model='openset', opt=True):
        self.num_points = num_points
        self.partition = partition
        self.data, self.label = load_data('train')

        num_ls = np.load('data_cls/ModelNet/100labeled_idx.npy')
        num_us = np.load('data_cls/ModelNet/100ulabeled_idx.npy')


        if self.partition == 'label':
            self.L_data = self.data[num_ls]
            self.L_label = self.label[num_ls]

            if len(idx)>0:
                self.L_data = self.L_data[idx]
                self.L_label = self.L_label[idx]
        else:
            self.U_data = self.data[num_us]
            self.U_label = self.label[num_us]

            self.add_data = np.load('data_cls/ModelNet/s3disadd03nadd01rotate_10000.npy')
            self.noise_data = np.load('data_cls/ModelNet/s3disnadd03nadd02rotate03drop05_10000.npy')


            if model == 'openset':
                self.U_data = np.concatenate((self.U_data, self.add_data, self.noise_data), axis=0)
            elif model == 'meta':
                self.U_data = self.U_data
            elif model == 'base':
                self.U_data = []
            else:
                base

    def __getitem__(self, item):

        if self.partition == 'label':
            pointcloud = self.L_data[item][:self.num_points]
            label = self.L_label[item]
            np.random.shuffle(pointcloud)
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            return pointcloud, label
        else:
            pointcloud = self.U_data[item][:self.num_points]
            np.random.shuffle(pointcloud)
            wa_pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            wa_pointcloud = translate_pointcloud(wa_pointcloud)
            
            np.random.shuffle(pointcloud)
            sa_pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            sa_pointcloud = translate_pointcloud(sa_pointcloud)
            sa_pointcloud = jitter_pointcloud(sa_pointcloud, sigma=0.01, clip=0.02)
            sa_pointcloud = random_mirror_pointcloud(sa_pointcloud)

            return wa_pointcloud, sa_pointcloud, item



    def __len__(self):
        if self.partition == 'label':
            return self.L_data.shape[0]
        else:
            return self.U_data.shape[0]


def load_shapenet_train():

    TRAINING_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/train_hdf5_file_list.txt'
    VAL_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/val_hdf5_file_list.txt'
    TESTING_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/test_hdf5_file_list.txt'
    h5_base_path = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data'

    ## train/val file list
    train_file_list = getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    val_file_list = getDataFiles(VAL_FILE_LIST)
    num_val_file = len(val_file_list)
    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)


    ## train/val file index
    train_file_idx = np.arange(0,num_train_file)
    val_file_idx = np.arange(0,num_val_file)
    test_file_idx = np.arange(0,num_test_file)

    ## Load Train Data
    train_data = []
    train_labels = []
    train_seg = []
    num_train = 0
    # train_data_idx = []
    for cur_train_filename in train_file_list:
        print('cur_train_filename',cur_train_filename)
        cur_train_data, cur_train_labels, cur_train_seg, cur_num_train, cur_train_data_idx = loadDataFile_with_seg(
            os.path.join(h5_base_path,cur_train_filename))



        train_data.append(cur_train_data)
        train_labels.append(cur_train_labels)
        train_seg.append(cur_train_seg)
        # train_data_idx.append(cur_train_data_idx+num_train)
        num_train += cur_num_train

    train_data = np.concatenate(train_data)

    train_labels = np.concatenate(train_labels).astype(np.int64)
    train_seg = np.concatenate(train_seg).astype(np.int64)
    
    # train_data_idx = np.concatenate(train_data_idx)
    num_train = num_train


    print('len(train_data)------',train_data.shape, train_data.dtype)
    print('len(train_labels)------',train_labels.shape, train_labels.dtype)
    print('len(train_seg)------',train_seg.shape, train_seg.dtype)
    # print('len(train_data_idx)------',train_data_idx.shape)
    print('num_train------',num_train)
    


    return train_data, train_labels, train_seg

def load_shapenet_val():

    TRAINING_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/train_hdf5_file_list.txt'
    VAL_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/val_hdf5_file_list.txt'
    TESTING_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/test_hdf5_file_list.txt'
    h5_base_path = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data'

    ## train/val file list
    train_file_list = getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    val_file_list = getDataFiles(VAL_FILE_LIST)
    num_val_file = len(val_file_list)
    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)


    ## train/val file index
    train_file_idx = np.arange(0,num_train_file)
    val_file_idx = np.arange(0,num_val_file)
    test_file_idx = np.arange(0,num_test_file)


    ## Load Val Data
    val_data = []
    val_labels = []
    val_seg = []
    num_val = 0
    # val_data_idx = []
    for cur_val_filename in val_file_list:
        cur_val_data, cur_val_labels, cur_val_seg, cur_num_val, cur_val_data_idx = loadDataFile_with_seg(
            os.path.join(h5_base_path, cur_val_filename))

        val_data.append(cur_val_data)
        val_labels.append(cur_val_labels)
        val_seg.append(cur_val_seg)
        # val_data_idx.append(cur_val_data_idx + num_val)
        num_val += cur_num_val

    val_data = np.concatenate(val_data)
    val_labels = np.concatenate(val_labels).astype(np.int64)
    val_seg = np.concatenate(val_seg).astype(np.int64)
    # val_data_idx = np.concatenate(val_data_idx)
    num_val = num_val


    print('len(val_data)------',val_data.shape)
    print('len(val_labels)------',val_labels.shape)
    print('len(val_seg)------',val_seg.shape)
    # print('len(val_data_idx)------',val_data_idx.shape)
    print('num_val------',num_val)
    

    return val_data, val_labels, val_seg

def load_shapenet_test():

    TRAINING_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/train_hdf5_file_list.txt'
    VAL_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/val_hdf5_file_list.txt'
    TESTING_FILE_LIST = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data/test_hdf5_file_list.txt'
    h5_base_path = '/data3/data4_lab-shi.xian/dataset/ShapeNet/hdf5_data'

    ## train/val file list
    train_file_list = getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    val_file_list = getDataFiles(VAL_FILE_LIST)
    num_val_file = len(val_file_list)
    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)


    ## train/val file index
    train_file_idx = np.arange(0,num_train_file)
    val_file_idx = np.arange(0,num_val_file)
    test_file_idx = np.arange(0,num_test_file)



    ## Load Test Data
    test_data = []
    test_labels = []
    test_seg = []
    num_test = 0
    # test_data_idx = []
    for cur_test_filename in test_file_list:
        print('cur_test_filename',cur_test_filename)
        cur_test_data, cur_test_labels, cur_test_seg, cur_num_test, cur_test_data_idx = load_h5_data_label_seg(
            os.path.join(h5_base_path,cur_test_filename))

        test_data.append(cur_test_data)
        test_labels.append(cur_test_labels)
        test_seg.append(cur_test_seg)
        # test_data_idx.append(cur_test_data_idx+num_test)
        num_test += cur_num_test


    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels).astype(np.int64)
    test_seg = np.concatenate(test_seg).astype(np.int64)
    # test_data_idx = np.concatenate(test_data_idx)
    num_test = num_test


    print('len(test_data)------',test_data.shape)
    print('len(test_labels)------',test_labels.shape)
    print('len(test_seg)------',test_seg.shape)
    # print('len(test_data_idx)------',test_data_idx.shape)
    print('num_test------',num_test)


    return test_data, test_labels, test_seg

class ShapeNet(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_points = num_points
        self.partition = partition        
        if self.partition == 'train':
            self.train_data, self.train_label, self.train_seg = load_shapenet_train()
        elif self.partition == 'val':
            self.val_data, self.val_label, self.val_seg = load_shapenet_val()
        elif self.partition == 'test':
            self.test_data, self.test_label, self.test_seg = load_shapenet_test()

    def __getitem__(self, item):

        if self.partition == 'train':
            train_data = self.train_data[item][:self.num_points]
            train_seg = self.train_seg[item][:self.num_points]
            train_label = self.train_label[item]

            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pc_idx = np.arange(self.num_points)
            # train_data = random_point_dropout(train_data) # open for dgcnn not for our idea  for all
            train_data = translate_pointcloud(train_data)
            np.random.shuffle(pc_idx)
            train_data = train_data[pc_idx]
            train_seg = train_seg[pc_idx]
            return train_data, train_seg, train_label, item

        elif self.partition == 'val':
            val_data = self.val_data[item][:self.num_points]
            val_seg = self.val_seg[item][:self.num_points]
            val_label = self.val_label[item]
            return val_data, val_seg, val_label

        elif self.partition == 'test':
            test_data = self.test_data[item][:self.num_points]
            test_seg = self.test_seg[item][:self.num_points]
            test_label = self.test_label[item]
            return test_data, test_seg, test_label

    def __len__(self):
        if self.partition == 'train':
            return self.train_data.shape[0]

        elif self.partition == 'val':
            return self.val_data.shape[0]

        elif self.partition == 'test':
            return self.test_data.shape[0]


class ShapeNet_US3DIS_StrAug(Dataset):
    def __init__(self, num_points, idx=[], partition='label', model='openset', opt=True):
        self.num_points = num_points
        self.partition = partition
        self.train_data, self.train_label, self.train_seg = load_shapenet_train()


        num_ls = np.load('data_seg/ShapeNet/100labeled_idx.npy')
        num_us = np.load('data_seg/ShapeNet/100ulabeled_idx.npy')
        
        if self.partition == 'label':
            self.L_data = self.train_data[num_ls]
            self.L_label = self.train_label[num_ls]
            self.L_seg = self.train_seg[num_ls]

            self.L_data = self.L_data[idx]
            self.L_label = self.L_label[idx]
            self.L_seg = self.L_seg[idx]

        else:
            self.U_data = self.train_data[num_us]
            self.U_label = self.train_label[num_us]
            self.U_seg = self.train_seg[num_us]

            self.add_data = np.load('data_cls/ModelNet/s3disadd03nadd01rotate_10000.npy')
            self.noise_data = np.load('data_cls/ModelNet/s3disnadd03nadd02rotate03drop05_10000.npy')


            if model == 'openset':
                self.U_data = np.concatenate((self.U_data, self.add_data, self.noise_data), axis=0)
            elif model == 'meta':
                self.U_data = self.U_data
            elif model == 'base':
                self.U_data = []
            else:
                base

            print('self.U_data', self.U_data.shape)
    def __getitem__(self, item):

        if self.partition == 'label':
            pointcloud = self.L_data[item][:self.num_points]
            label = self.L_label[item]
            seg = self.L_seg[item]
            np.random.shuffle(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            return pointcloud, seg, label
        else:
            pointcloud = self.U_data[item][:self.num_points]
            np.random.shuffle(pointcloud)
            wa_pointcloud = random_point_dropout(pointcloud) 
            wa_pointcloud = translate_pointcloud(wa_pointcloud)
            
            np.random.shuffle(pointcloud)
            sa_pointcloud = random_point_dropout(pointcloud)
            sa_pointcloud = translate_pointcloud(sa_pointcloud)
            sa_pointcloud = jitter_pointcloud(sa_pointcloud, sigma=0.01, clip=0.02)
            sa_pointcloud = random_mirror_pointcloud(sa_pointcloud)

            return wa_pointcloud, sa_pointcloud, item



    def __len__(self):
        if self.partition == 'label':
            return self.L_data.shape[0]
        else:
            return self.U_data.shape[0]

def load_PerCat_train():

    f = h5py.File('data/percat_chair/chair_train.h5', 'r')
    tp_pts = np.array(f['ed_point'])
    tp_label = np.array(f['ed_label'], dtype=np.int64)
    tg_pts = np.array(f['un_point'])
    tg_label = np.array(f['un_label'], dtype=np.int64)
    f.close()

    idx_list = np.load('data/percat_chair/tp_idx_list.npy')

    print('train tp_pts', tp_pts.shape, tp_pts.dtype)
    print('train tp_label', tp_label.shape, tp_label.dtype, np.unique(tp_label))
    print('train tg_pts', tg_pts.shape, tg_pts.dtype)
    print('train tg_label', tg_label.shape, tg_label.dtype, np.unique(tg_label))
    print('train idx_list', idx_list.shape, idx_list.dtype)


    min_label = np.min(tg_label)
    tg_label = tg_label - min_label
    tp_label = tp_label - min_label

    return tg_pts, tg_label, tp_pts, tp_label, idx_list

def load_PerCat_test():

    f = h5py.File('data/percat_chair/chair_test.h5', 'r')
    tp_pts = np.array(f['template_point'])
    tp_label = np.array(f['template_label'], dtype=np.int64)
    tg_pts = np.array(f['target_point'])
    tg_label = np.array(f['target_label'], dtype=np.int64)
    f.close()

    print('test tp_pts', tp_pts.shape, tp_pts.dtype)
    print('test tp_label', tp_label.shape, tp_label.dtype, np.unique(tp_label))
    print('test tg_pts', tg_pts.shape, tg_pts.dtype)
    print('test tg_label', tg_label.shape, tg_label.dtype, np.unique(tg_label))

    min_label = np.min(tg_label)
    tg_label = tg_label - min_label
    tp_label = tp_label - min_label

    return tg_pts, tg_label, tp_pts, tp_label

def load_Semi_train():

    wl_tg_pts = []
    wl_tg_label = []
    wl_shape_label = []
    wl_tp_pts = []
    wl_tp_label = []
    wl_idx_list = []
    temp_idx = 0
    for i_label in range(16):

        f = h5py.File('data/' + np.str(i_label) + '/train_data.h5', 'r')
        tp_pts = np.array(f['ed_point'])
        tp_label = np.array(f['ed_label'], dtype=np.int64)
        tg_pts = np.array(f['un_point'])
        tg_label = np.array(f['un_label'], dtype=np.int64)
        f.close()


        idx_list = np.load('data/' + np.str(i_label) + '/tp_idx_list.npy')
        # idx_list = idx_list + temp_idx
        # temp_idx = temp_idx + tp_pts.shape[0]
        # print('temp_idx', temp_idx)

        wl_tg_pts.append(tg_pts)
        wl_tg_label.append(tg_label)

        for i in range(tg_pts.shape[0]):
            wl_tp_pts.append(tp_pts[idx_list[i, 0]].reshape(1, 2048, 3))
            wl_tp_label.append(tp_label[idx_list[i, 0]].reshape(1, 2048))

        wl_shape_label.append(np.ones((idx_list.shape[0], 1), dtype=np.int64) * i_label)

    wl_tg_pts = np.concatenate(wl_tg_pts)
    wl_tg_label = np.concatenate(wl_tg_label)
    wl_tp_pts = np.concatenate(wl_tp_pts)
    wl_tp_label = np.concatenate(wl_tp_label)
    # wl_idx_list = np.concatenate(wl_idx_list)
    wl_shape_label = np.concatenate(wl_shape_label)

    print('train tp_pts', wl_tp_pts.shape, wl_tp_pts.dtype)
    print('train tp_label', wl_tp_label.shape, wl_tp_label.dtype)
    print('train tg_pts', wl_tg_pts.shape, wl_tg_pts.dtype)
    print('train tg_label', wl_tg_label.shape, wl_tg_label.dtype)
    # print('train idx_list', wl_idx_list.shape, wl_idx_list.dtype, np.min(wl_idx_list), np.max(wl_idx_list))
    print('train shape_label', wl_shape_label.shape, wl_shape_label.dtype, np.unique(wl_shape_label))
    
    return wl_tg_pts, wl_tg_label, wl_tp_pts, wl_tp_label, wl_shape_label

def load_Semi_test():

    wl_tg_pts = []
    wl_tg_label = []
    wl_shape_label = []
    wl_tp_pts = []
    wl_tp_label = []
    for i_label in range(16):

        f = h5py.File('data/' + np.str(i_label) + '/test_data.h5', 'r')
        tp_pts = np.array(f['template_point'])
        tp_label = np.array(f['template_label'], dtype=np.int64)
        tg_pts = np.array(f['target_point'])
        tg_label = np.array(f['target_label'], dtype=np.int64)
        f.close()

        wl_tg_pts.append(tg_pts)
        wl_tg_label.append(tg_label)
        wl_tp_pts.append(tp_pts)
        wl_tp_label.append(tp_label)

        wl_shape_label.append(np.ones((tp_pts.shape[0], 1), dtype=np.int64) * i_label)


    wl_tg_pts = np.concatenate(wl_tg_pts)
    wl_tg_label = np.concatenate(wl_tg_label)
    wl_tp_pts = np.concatenate(wl_tp_pts)
    wl_tp_label = np.concatenate(wl_tp_label)

    wl_shape_label = np.concatenate(wl_shape_label)

    print('test tp_pts', wl_tp_pts.shape, wl_tp_pts.dtype)
    print('test tp_label', wl_tp_label.shape, wl_tp_label.dtype)
    print('test tg_pts', wl_tg_pts.shape, wl_tg_pts.dtype)
    print('test tg_label', wl_tg_label.shape, wl_tg_label.dtype)
    print('test shape_label', wl_shape_label.shape, wl_shape_label.dtype, np.unique(wl_shape_label))

    return wl_tg_pts, wl_tg_label, wl_tp_pts, wl_tp_label, wl_shape_label

#USE For SUNCG, to center to origin
def center_data(pcs):
	for pc in pcs:
		centroid = np.mean(pc, axis=0)
		pc[:,0]-=centroid[0]
		pc[:,1]-=centroid[1]
		pc[:,2]-=centroid[2]
	return pcs

def normalize_data(pcs):
	for pc in pcs:
		#get furthest point distance then normalize
		d = max(np.sum(np.abs(pc)**2,axis=-1)**(1./2))
		pc /= d

		# pc[:,0]/=max(abs(pc[:,0]))
		# pc[:,1]/=max(abs(pc[:,1]))
		# pc[:,2]/=max(abs(pc[:,2]))

	return pcs

class ScanObject(Dataset):
    def __init__(self, num_points, partition='train',bg=True):
        # self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition      


        if self.partition == 'train':
            if bg:
                f = h5py.File('/data3/data4_lab-shi.xian/dataset/scanobject/main_split/training_objectdataset.h5')
                # f = h5py.File('/data3/data4_lab-shi.xian/dataset/scanobject/main_split/training_objectdataset_augmentedrot_scale75.h5')
                self.data = f['data'][:].astype('float32')
                self.label = f['label'][:].astype('int64')
                self.seg = f['mask'][:].astype('int64')
                # for key in f.keys():
                #     print('key', key)
                f.close()
        elif self.partition == 'test':
            if bg:
                f = h5py.File('/data3/data4_lab-shi.xian/dataset/scanobject/main_split/test_objectdataset.h5')
                # f = h5py.File('/data3/data4_lab-shi.xian/dataset/scanobject/main_split/test_objectdataset_augmentedrot_scale75.h5')
                self.data = f['data'][:].astype('float32')
                self.label = f['label'][:].astype('int64')
                self.seg = f['mask'][:].astype('int64')
                # for key in f.keys():
                #     print('key', key)
                f.close()
        # self.data = np.concatenate(self.data, axis=0)
        # self.label = np.concatenate(self.label, axis=0)
        # self.seg = np.concatenate(self.seg, axis=0)
        self.data = center_data(self.data)
        self.data = normalize_data(self.data)
        self.label = self.label.reshape(-1, 1)
        print('self.data', self.data.shape)
        print('self.label', self.label.shape, np.unique(self.label))
        print('self.seg', self.seg.shape, np.unique(self.seg))


        # all_data = []
        # all_label = []
        # for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        #     f = h5py.File(h5_name)
        #     data = f['data'][:].astype('float32')
        #     label = f['label'][:].astype('int64')
        #     f.close()
        #     all_data.append(data)
        #     all_label.append(label)
        # all_data = np.concatenate(all_data, axis=0)
        # all_label = np.concatenate(all_label, axis=0)
        # print('all_data', all_data.shape)
        # print('all_label', all_label.shape)



    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]




if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)


    train = ShapeNet(2048, 'train')
    val = ShapeNet(2048, 'val')
    test = ShapeNet(2048, 'test')
    for data, seg, label in train:
        print(data.shape)
        print(seg.shape)
        print(label.shape)

