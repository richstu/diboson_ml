#! /usr/bin/env python3

import math,os,sys, datetime
import numpy as np
import pandas as pd
# to force running on CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import argparse
import utils
from coffea import hist
from termcolor import colored

from eval_dnn import eval_dnn

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='ML fun with higgsinos.')
  parser.add_argument('--cpu', help='Use cpu', action='store_true')
  parser.add_argument('-i','--input_path', help='Path to training data.',
                      default='/net/cms29/cms29r0/atto/v2/2016/raw_atto/train_raw_atto_TChiHH_HToBB_HToBB_3D_2016.root')
  parser.add_argument('-t','--tag', help='Any tag to add to output filenames.',default='default')
  parser.add_argument('-e','--epochs', type=int, help='Number of epochs',default=400)
  parser.add_argument('-d','--dense', type=int, help='Number of dense layers',default=4)
  parser.add_argument('-n','--nodes', type=int, help='Number of nodes per dense layer.',default=400)
  parser.add_argument('--activation', help='Activation function',default='relu')
  parser.add_argument('--loss', help='Loss function.',default='mean_squared_error')
  parser.add_argument('--optimizer', help='Optimizer.',default='adam')
  parser.add_argument('--val_frac', type=float, help='Fraction of data to use for calculating validation loss.',default=0.2)
  parser.add_argument('--random_data', action='store_true', 
                      help='Generated gaussian distributed random numbers as input for the training')
  parser.add_argument('--log_transform', action='store_true', 
                      help='Take the log of jet_pt, jet_m, mjj, mhiggs, i.e. all variables in GeV as of now.')
  args = parser.parse_args()

  device = "CPU" if args.cpu else "GPU"
  tf.device('/'+device+':0')

  t0 = time()
  np.set_printoptions(precision=2, suppress=True)
  print('Reading data: '+colored(args.input_path,'yellow'))
  x_train, y_train, mh_mean_train, mh_std_train = utils.get_data(args.input_path, args.log_transform)
  if args.random_data:
    x_train = np.random.randn(x_train.shape[0],x_train.shape[1])
  print('\nTook %.0f seconds to prepare data.' % (time()-t0))

  model = utils.define_sequential_model(x_train.shape[1], args.dense, args.nodes, args.loss, args.optimizer, args.activation)
  
  # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  history = model.fit(x_train, y_train, 
                      epochs=int(args.epochs), validation_split=args.val_frac,
                      batch_size=len(y_train))
                      # callbacks=[tensorboard_callback])

  model_name = args.tag+'_arc-%ix%i_lay-%s_opt-%s_act-%s_epo-%i' % (args.dense, args.nodes, args.loss, args.optimizer, args.activation, args.epochs)
  if args.log_transform:
    model_name += '_log'
  model_name += '_hmean-%.3f_hstd-%.3f' % (mh_mean_train, mh_std_train)
  model_name = model_name.replace('.','p')
  model.save(model_name+'.h5')

  pd.DataFrame(history.history).plot(figsize=(8, 5)) 
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.savefig('history_'+model_name+'.pdf')

  # might as well also make the plots since it doesn't take long
  test_data_path = args.input_path.replace('train','test')
  eval_dnn(model, model_name.replace('.h5',''), test_data_path, mh_mean_train, mh_std_train, args.log_transform)

  print('\nProgram took %.0fm %.0fs.' % ((time()-t0)/60,(time()-t0)%60))
