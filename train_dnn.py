#! /usr/bin/env python3

import math,os,sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import argparse
import utils
from coffea import hist

from eval_utils import eval_model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='ML fun with higgsinos.')
  parser.add_argument('--cpu', help='Use cpu', action='store_true')
  parser.add_argument('-e','--epochs', type=int, help='Number of epochs',default=10)
  parser.add_argument('-d','--dense', type=int, help='Number of dense layers',default=4)
  parser.add_argument('-n','--nodes', type=int, help='Number of nodes per dense layer.',default=400)
  parser.add_argument('-a','--activation', help='Activation function',default='relu')
  parser.add_argument('-l','--loss', help='Loss function.',default='mean_squared_error')
  parser.add_argument('-o','--optimizer', help='Optimizer.',default='adam')
  parser.add_argument('-t','--val_frac', type=float, help='Fraction of data to use for calculating validation loss.',default=0.2)
  parser.add_argument('--random_data', action='store_true', help='Generated gaussian distributed random numbers as input for the training')
  args = parser.parse_args()

  device = "CPU" if args.cpu else "GPU"
  tf.device('/'+device+':0')

  t0 = time()
  np.set_printoptions(precision=2, suppress=True)
  path = ''
  if os.getenv('HOSTNAME'):
    path = '/net/cms29' 
  path +='/cms29r0/atto/v1/2016/raw_atto/train_raw_atto_TChiHH_HToBB_HToBB_3D_2016.root'
  print('Reading data: '+path)
  x_train, y_train, mh_mean_train, mh_std_train = utils.get_data(path)
  # if args.random_data:
  #   x_train = utils.get_random_data(path, x_train.shape[0],x_train.shape[1])  
  print('\nTook %.0f seconds to prepare data.' % (time()-t0))

  model = utils.define_sequential_model(x_train.shape[1], args.dense, args.nodes, args.loss, args.optimizer, args.activation)
  history = model.fit(x_train, y_train, epochs=int(args.epochs), validation_split=args.val_frac)

  model_name = 'seq_arc-%ix%i_lay-%s_opt-%s_act-%s_epo-%i' % (args.dense, args.nodes, args.loss, args.optimizer, args.activation, args.epochs)
  model_name += '_hmean-%.3f_hstd-%.3f' % (mh_mean_train, mh_std_train)
  model_name = model_name.replace('.','p')
  model.save(model_name+'.h5')

  pd.DataFrame(history.history).plot(figsize=(8, 5)) 
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.savefig('history_'+model_name+'.pdf')

  # might as well also make the plots since it doesn't take long
  test_data_path = ''
  if os.getenv('HOSTNAME'):
    test_data_path = '/net/cms29' 
  test_data_path +='/cms29r0/atto/v1/2016/raw_atto/test_raw_atto_TChiHH_HToBB_HToBB_3D_2016.root'

  eval_model(model, test_data_path, mh_mean_train, mh_std_train)

  print('\nProgram took %.0f:%.0f.' % ((time()-t0)/60,(time()-t0)%60))
