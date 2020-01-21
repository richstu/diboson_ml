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
    model_.add(keras.layers.Dense(nodes, activation=activation))
  model_.add(keras.layers.Dense(1))
  model_.compile(loss=loss, optimizer=optimizer)
  model_.summary()
  return model_

def fit_gauss(xi, yi, verbose=False):
  sigma = math.sqrt(xi.var())
  init  = [1./sigma/math.sqrt(2*math.pi), xi.mean(), sigma, 0.01]
  fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
  errfunc  = lambda p, x, y: (y - fitfunc(p, x))
  fit_result, fit_cov  = leastsq( errfunc, init, args=(xi, yi))
  if (verbose):
    print("Fit results: mean = %.2f, sigma = %.2f" % (fit_result[1],fit_result[2]))
  return fitfunc(fit_result, xi)

def stack_feats(norm_tree, feat_names, max_length, dummy = -9):
  verbose = False
  # pad the number of objects for each feature, so that there is always 5 objects, filling with 9's
  # to use just the top 4 jets -> change max_length to 4 and clip to True (N.B. global features would require dropping things too...)
  padded_data = norm_tree[feat_names].pad(max_length,axis=0,clip=False).fillna(dummy)
  if verbose: pprint(padded_data.contents)
  # then, convert ordered dict of ChunkedArray to numpy 3D array: axis 0 = ith feature; axis 1 = ith event, axis 2 = ith jet
  arr_3d = np.asarray([padded_data[ifeat].regular() for ifeat in feat_names])
  # then, stack features so now arr_3d is interpreted as a list of 2D arrays with axis 0 = ith event, axis 1 = ith jet
  # output is then axis 0 = ith event, axis 1 = ith jet, axis 2 = ith feature
  arr_3d = np.stack(arr_3d, axis=2)
  # finally, reshape to concatenate features for all jets into 1D array (axis 0 remains = ith event)
  n0_,n1_,n2_ = arr_3d.shape
  return arr_3d.reshape(n0_,n1_*n2_)

def get_data(path):
  nobj = 5 # this can't be changed without remaking atto ntuples for now
  obj_feats = ['jet_pt','jet_eta','jet_phi','jet_m','jet_deepcsv']
  nobj_feats = len(obj_feats)

  glob_feats = ['mjj','drjj']
  ncombinations = int(math.factorial(nobj)/(2*math.factorial(nobj-2)))

  branches = ['mhiggs']
  branches.extend(obj_feats)
  branches.extend(glob_feats)

  tree = uproot.tree.lazyarrays(path=path, treepath='tree', branches=branches, namedecode='utf-8')
  print('Found %i entries' % len(tree))
  # print(tree.contents)

  mh_mean, mh_std = 0, 0 
  for col in tree.columns:
    flatter = np.asarray(tree[col].flatten())
    mean = flatter.mean()
    std = flatter.std()
    if 'mhiggs' in col: # save those to convert output later
      mh_mean = mean
      mh_std = std
    print('Fearture %s has mean = %.2f and std = %.2f' % (col, mean, std))
    tree[col] = tree[col] - mean
    tree[col] = tree[col]*(1/std)

  obj_feats_data = stack_feats(tree, obj_feats, nobj)
  glob_feats_data = stack_feats(tree, glob_feats, ncombinations)

  x_data_ = np.concatenate([obj_feats_data, glob_feats_data], axis=1)
  y_data_ = np.asarray(tree['mhiggs'])

  return x_data_, y_data_, mh_mean, mh_std
