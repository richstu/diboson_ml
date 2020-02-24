#! /usr/bin/env python3
import os,datetime,sys
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
  parser.add_argument('--in_sig', help='Path to training signal sample.',
                      default='/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/dnn_mc/TChiHH/higfeats_preselect/higfeats_merged_higmc_preselect_SMS-TChiHH_HToBB_HToBB_3D_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISummer16NanoAODv5.root')
  parser.add_argument('--in_bkg', help='Path to training background sample.',
                      default='/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/dnn_mc/mc/higfeats_preselect/higfeats_merged_pico_preselect_higloose_met150_TTJets_SingleLeptFromT_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_higmc_higloose_nfiles_28.root')
  parser.add_argument('-t','--tag', help='Any tag to add to output filenames.',default='MLP')
  parser.add_argument('-e','--epochs', type=int, help='Number of epochs',default=30)
  parser.add_argument('-d','--dense', type=int, help='Number of dense layers',default=5)
  parser.add_argument('-n','--nodes', type=int, help='Number of nodes per dense layer.',default=200)
  parser.add_argument('--activation', help='Activation function',default='elu')
  parser.add_argument('--loss', help='Loss function.',default='mean_absolute_error')
  parser.add_argument('--optimizer', help='Optimizer.',default='adam')
  parser.add_argument('--val_frac', type=float, help='Fraction of data to use for calculating validation loss.',default=0.2)
  parser.add_argument('--train_bkg', action='store_true', help='Use a mix of background and signal for the training')   
  parser.add_argument('--log_transform', action='store_true', 
                      help='Take the log of jet_pt, jet_m, mjj, mhiggs, i.e. all variables in GeV as of now.')
  args = parser.parse_args()

  device = "CPU" if args.cpu else "GPU"
  tf.device('/'+device+':0')

  t0 = time()
  np.set_printoptions(precision=2, suppress=True)
  # np.set_printoptions(threshold=sys.maxsize)
  print('Reading data: '+colored(args.in_sig,'yellow'))

  model_name = args.tag+'-%ix%i_%s_%s_%s_e%i' % (args.dense, args.nodes, args.loss, args.optimizer, args.activation, args.epochs)
  if args.log_transform:
    model_name += '_log'
  model_name = model_name.replace('.','p')
  print('Model name set to: '+colored(model_name,'green'))

  if args.train_bkg:
    x_train, y_train = utils.get_data(model_name=model_name, path=args.in_sig, extra_path=args.in_bkg, do_log=args.log_transform, training=True, nent=-1)
  else:
    x_train, y_train = utils.get_data(model_name=model_name, path=args.in_sig, extra_path='', do_log=args.log_transform, training=True, nent=-1)

  rng_state = np.random.get_state()
  np.random.shuffle(x_train)
  np.random.set_state(rng_state)
  np.random.shuffle(y_train)

  print('\nTook %.0f seconds to prepare data.' % (time()-t0))

  model = model_defs.mlp(x_train.shape[1], args.dense, args.nodes, args.loss, args.optimizer, args.activation)
  
  # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  history = model.fit(x_train, y_train, 
                      epochs=int(args.epochs), validation_split=args.val_frac,
                      batch_size=1024)
                      # callbacks=[tensorboard_callback])

  model.save(model_name+'.h5')

  hdf = pd.DataFrame(history.history)
  hdf.plot(figsize=(8, 5)) 
  plt.grid(True)
  # plt.gca().set_ylim()
  plt.savefig('history_'+model_name+'.pdf')
  print('imgcat','history_'+model_name+'.pdf')

  print('\nProgram took %.0fm %.0fs.' % ((time()-t0)/60,(time()-t0)%60))
