#! /usr/bin/env python3

import math
import numpy as np
import uproot
from coffea import hist
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import tensorflow.keras as keras
from pprint import pprint

def define_sequential_model(nfeats, dense_layers, nodes, loss, optimizer, activation):
  model_ = keras.models.Sequential()
  model_.add(keras.layers.Input(shape=(nfeats)))
  for i in range(dense_layers):
    model_.add(keras.layers.Dense(nodes, activation=activation, kernel_initializer="lecun_normal"))
  model_.add(keras.layers.Dense(1))
  model_.compile(loss=loss, optimizer=optimizer)
  model_.summary()
  return model_

def get_data(path, read_mhiggs=True, do_log=False, dummy = -9):
  nobj = 5 # this can't be changed without remaking atto ntuples for now
  obj_feats = [b'jet_brank_pt',b'jet_brank_eta',b'jet_brank_phi',b'jet_brank_m',b'jet_brank_deepcsv']
  ncombinations = int(math.factorial(nobj)/(2*math.factorial(nobj-2)))
  glob_feats = [b'mjj',b'drjj']
  nfeats = nobj*len(obj_feats) + ncombinations*len(glob_feats)

  branches = []
  if read_mhiggs: branches.append(b'mhiggs')
  branches.extend(obj_feats)
  branches.extend(glob_feats)
  tree = uproot.open(path=path)['tree']
  print('Found %i entries' % len(tree))
  awk_arrays = tree.arrays(branches)

  # create array with the final size to reduce copying
  x_data_ = np.empty(shape=(len(tree),nfeats))

  # normalize all features, saving the normalization parameters for the higgs mass in order to transform the output
  mh_mean, mh_std = 0, 0 
  for branch in branches:
    # print('Processing branch:',branch)
    if do_log and col in [b'jet_brank_pt',b'jet_brank_m',b'mjj']:
      awk_arrays[branch] = np.log(awk_arrays[branch])
    # flatten to compute mean and std
    flat_arr = awk_arrays[branch].flatten()
    mean, std = flat_arr.mean(), flat_arr.std()
    awk_arrays[branch] = awk_arrays[branch] - mean
    awk_arrays[branch] = awk_arrays[branch]*(1/std)
    if b'mhiggs' in branch: # save those to convert output later
      mh_mean, mh_std = mean, std
    # print('Fearture %s has mean = %.2f and std = %.2f' % (branch, mean, std))

  # shape up jet features data
  for i, branch in enumerate(obj_feats):
    # pad the jet arrays so all events have 5 jets
    awk_arrays[branch] = awk_arrays[branch].pad(nobj,axis=0,clip=True).fillna(dummy)
    # convert to 2D numpy array
    awk_arrays[branch] = awk_arrays[branch].regular()
    # fill the jet data in the final array such that the order is j1_pt, j1_eta, ..., j1_csv, j2_pt, j2_eta...
    # by assigning the array for each jet feature to an interleaved view of the large x_data_ array
    # this will fill up the x_data array until nobj*len(obj_feats) along axis 1
    x_data_[:,i:nobj*len(obj_feats):len(obj_feats)] = awk_arrays[branch]

  # shape up global features data
  for i, branch in enumerate(glob_feats):
    awk_arrays[branch] = awk_arrays[branch].pad(ncombinations,axis=0,clip=True).fillna(dummy)
    awk_arrays[branch] = awk_arrays[branch].regular()
    # fill global data starting after the jet data, i.e at nobj*len(obj_feats) along axis 1
    x_data_[:,nobj*len(obj_feats)+i::len(glob_feats)] = awk_arrays[branch]
  
  y_data_ = None
  if (read_mhiggs): awk_arrays[b'mhiggs']
  return x_data_, y_data_, mh_mean, mh_std
