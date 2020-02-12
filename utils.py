#! /usr/bin/env python3

import math, json
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

def get_data(path, do_log=False, training = False, dummy = -9):
  nobj = 5 # this can't be changed without remaking higfeats ntuples since mjj also has to change
  obj_feats = [b'jet_brank_pt',b'jet_brank_eta',b'jet_brank_phi',b'jet_brank_m',b'jet_brank_deepcsv']
  ncombinations = int(math.factorial(nobj)/(2*math.factorial(nobj-2)))
  glob_feats = [b'mjj',b'drjj']
  nfeats = nobj*len(obj_feats) + ncombinations*len(glob_feats)

  branches = []
  if training: 
    branches.append(b'mhiggs')
  branches.extend(obj_feats)
  branches.extend(glob_feats)
  tree = uproot.open(path=path)['tree']
  print('Found %i entries' % len(tree))
  awk_arrays = tree.arrays(branches)

  # cut off the threshold peak at 50 due to Pythia minimum mass
  mask = None
  if (training): 
    mask = awk_arrays[b'mhiggs']>51 

  # create array with the final size to reduce copying
  rows = len(tree)
  if (training): 
    rows = np.count_nonzero(mask)
  x_data_ = np.empty(shape=(rows,nfeats))

  # normalize all features, saving the normalization parameters for the higgs mass in order to transform the output
  norm_dict = {}
  if not training:
    with open("norm_dict.json","r") as fin:
      norm_dict = json.load(fin)

  for branch in branches:
    if mask is not None:
      awk_arrays[branch] = awk_arrays[branch][mask]
    # do not normalize output
    if b'mhiggs' in branch:
      continue
    # print('Processing branch:',branch)
    if do_log and branch in [b'jet_brank_pt',b'jet_brank_m',b'mjj']:
      awk_arrays[branch] = np.log(awk_arrays[branch])
    # flatten to compute mean and std
    mean, std = 0, 0
    if training:
      flat_arr = awk_arrays[branch].flatten()
      mean, std = flat_arr.mean(), flat_arr.std()
      norm_dict[branch.decode()] = (str(mean), str(std))
    else:
      mean, std = float(norm_dict[branch.decode()][0]), float(norm_dict[branch.decode()][1])
    awk_arrays[branch] = awk_arrays[branch] - mean
    awk_arrays[branch] = awk_arrays[branch]*(1/std)
    # if b'mhiggs' in branch: # save those to convert output later
    #   mh_mean, mh_std = mean, std
    print('Fearture %s has mean = %.2f and std = %.2f' % (branch, mean, std))

  if training:
    with open("norm_dict.json","w") as fout:
      fout.write(json.dumps(norm_dict))

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
  if training: 
    y_data_ = awk_arrays[b'mhiggs']

  return x_data_, y_data_
