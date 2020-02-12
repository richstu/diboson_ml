#! /usr/bin/env python3
import os,datetime
from time import time
import numpy as np
# to force running on CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorflow.keras as keras
import argparse
import utils, model_defs
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored, cprint

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='ML fun with higgsinos.')
  parser.add_argument('--cpu', help='Use cpu', action='store_true')
  parser.add_argument('-i','--input_path', help='Path to training data.',
                      default='/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/dnn_mc/higfeats_unskimmed/higfeats_raw_pico_SMS-TChiHH_HToBB_HToBB_3D_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISummer16NanoAODv5__PUMoriond17_Nano1June2019_102X_mcRun2_asymptotic_v7_train.root')
  parser.add_argument('-t','--tag', help='Any tag to add to output filenames.',default='MLP')
  parser.add_argument('-e','--epochs', type=int, help='Number of epochs',default=30)
  parser.add_argument('-d','--dense', type=int, help='Number of dense layers',default=5)
  parser.add_argument('-n','--nodes', type=int, help='Number of nodes per dense layer.',default=200)
  parser.add_argument('--activation', help='Activation function',default='elu')
  parser.add_argument('--loss', help='Loss function.',default='mean_absolute_error')
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
  x_train, y_train = utils.get_data(args.input_path, args.log_transform, training=True)
  if args.random_data:
    x_train = np.random.randn(x_train.shape[0],x_train.shape[1])
  print('\nTook %.0f seconds to prepare data.' % (time()-t0))

  model = model_defs.mlp(x_train.shape[1], args.dense, args.nodes, args.loss, args.optimizer, args.activation)
  
  # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  history = model.fit(x_train, y_train, 
                      epochs=int(args.epochs), validation_split=args.val_frac,
                      batch_size=1024)
                      # callbacks=[tensorboard_callback])

  model_name = args.tag+'%ix%i_%s_%s_%s_e%i' % (args.dense, args.nodes, args.loss, args.optimizer, args.activation, args.epochs)
  if args.log_transform:
    model_name += '_log'
  model_name = model_name.replace('.','p')
  model.save(model_name+'.h5')

  pd.DataFrame(history.history).plot(figsize=(8, 5)) 
  plt.grid(True)
  plt.gca().set_ylim(20,50)
  plt.savefig('history_'+model_name+'.pdf')
  print('imgcat','history_'+model_name+'.pdf')

  print('\nProgram took %.0fm %.0fs.' % ((time()-t0)/60,(time()-t0)%60))
