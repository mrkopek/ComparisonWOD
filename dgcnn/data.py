#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
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
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class WashingtonDataset(Dataset):
    def __init__(self, partition,trail):
        self.data, self.label = load_rgbd_data(trail,partition=="train")
        self.num_points = 256
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    

def load_rgbd_data(trail,istrain):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data\\rgbd_data')
    data_x=np.loadtxt("rgbd_subset_256_x")
    data_y=np.loadtxt("rgbd_subset_256_y")
    data_z=np.loadtxt("rgbd_subset_256_z")
    label=np.loadtxt("rgbd_subset_256_labelc")
    instancelabel=pd.read_csv("rgbd_subset_256_labeli",header=None).transpose()
    testinstances=pd.read_csv("testinstances2.csv",delimiter=";",header=None).transpose()
    train_data = np.array([data_x,data_y,data_z])
    train_data=train_data.transpose(1,2,0)

    this_testinstance=testinstances.values[trail]
    label = label-1
    label_shrink=np.isin(label,range(51))
    label=label[label_shrink]
    train_data=train_data[label_shrink]
    
    label = torch.Tensor(label).type(torch.int64)
    
    label=label.transpose(-1,0)
    
    test_instancelabel =instancelabel.isin(this_testinstance).values
    test_instancelabel = test_instancelabel.flatten()
    
    test_instancelabel=test_instancelabel[label_shrink]

    train_data=train_data.astype("float32")
    test_data = train_data [test_instancelabel]
    train_data = train_data[~test_instancelabel]
    test_label = label[test_instancelabel]
    train_label = label[~test_instancelabel]
    if istrain:
        return train_data,train_label
    else:
        return test_data,test_label


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)