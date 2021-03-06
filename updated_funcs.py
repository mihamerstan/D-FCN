import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import matplotlib.pylab as plt
import os
import pickle
# os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']="1"

import sys
BASE_DIR = os.path.abspath('')
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'tf_utils'))
import provider
import tf_util


def pc_normalize_min(data):
    mindata = np.min(data[:,:3], axis=0)
    # MS: Function now returns mindata here
    return (data[:,:3] - mindata), mindata

def get_batch_wdp(dataset, batch_idx):
    bsize = BATCH_SIZE
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    ps_xyz_min = np.zeros((bsize, 1, 3))
    ps_out = np.zeros((bsize, NUM_POINT, 3))

    batch_feats = np.zeros((bsize, NUM_POINT, 1))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw,feat = get_batch('train',index=0)
        # ps_out collects the pre-normalized batch_data
        ps_out[i,...] = ps
        # ps_min is collected not from pc_normalize_min
        ps,ps_min = pc_normalize_min(ps)
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
        batch_feats[i,:] = feat

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
    	# ps_xyz_min collects the xyz minimums vector for each batch
        ps_xyz_min[i,...] = ps_min
    # ps_xyz_min and px_out are now returned.
    return batch_data, batch_label, batch_smpw, batch_feats, ps_xyz_min, ps_out

def main():
	batch_data, batch_label, batch_smpw, batch_feats, ps_xyz_min, ps_out = get_batch_wdp('train', k)
	# You don't need both ps_out and ps_xyz_min, but I returned both to confirm that the math works
	# The following print-outs should be the same, and so you should be able 
	# Subtract ps_xyz_min from the xyz coords of the output of eval_one_epoch_whole_scene() 
	# And get the original xyz coordinates.
	print(batch_data[0][0])
	print(ps_out[0][0]-ps_xyz_min[0])