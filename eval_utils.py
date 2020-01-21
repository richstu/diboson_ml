#! /usr/bin/env python3

import math,os,sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import uproot
from pprint import pprint
from time import time
import argparse
import utils
from coffea import hist
from glob import glob

def eval_model(model, tag, path, mh_mean_train, mh_std_train):
  print('Reading test data: '+path)

  x_test, y_test, mh_mean_test, mh_std_test = utils.get_data(path)
  y_test = y_test*mh_std_test + mh_mean_test

  y_pred = model.predict(x_test).flatten()
  y_pred = y_pred*mh_std_train + mh_mean_train
  sigma_dnn = y_test - y_pred

  # Read some extra branches for evaluation studies
  branches = ['hig_am','nbacc','mchi','mlsp']
  tree = uproot.tree.lazyarrays(path=path, treepath='tree', branches=branches, namedecode='utf-8')
  y_cb = np.asarray(tree['hig_am'])
  sigma_cb = y_test - y_cb

  hsigma = hist.Hist("Events",
                      hist.Cat("method", "Reco method"),
                      hist.Bin("sigma", "True - Predicted", 200, -300, 300))
  hsigma.fill(method="DNN", sigma=sigma_dnn)
  hsigma.fill(method="CB", sigma=sigma_cb)

  fig = plt.figure()
  ax = hist.plot1d(hsigma, overlay="method", stack=False)
  # fig, ax, __ = hist.plot1d(hsigma, overlay="method", stack=False)
  # gauss = utils.fit_gauss(hsigma.axis('sigma').centers(), hsigma.values()[('DNN',)], True)
  # ax.plot(hsigma.axis('sigma').centers(), gauss, color='green', linewidth=2.5, label=r'Fitted function')
  fig.savefig('sigma_'+tag+'.pdf')

  fig = plt.figure()
  hmh = hist.Hist("Events",
                      hist.Cat("method", "Reco method"),
                      hist.Bin("mhiggs", "Higgs mass", 30, 0, 300))
  hmh.fill(method="Generated", mhiggs=y_test)
  hmh.fill(method="DNN", mhiggs=y_pred)
  hmh.fill(method="CB", mhiggs=y_cb)

  ax = hist.plot1d(hmh, overlay="method", stack=False)
  # fig, ax, __ = hist.plot1d(hmh, overlay="method", stack=False)
  fig.savefig('mh_'+tag+'.pdf')
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate model performance.')
  parser.add_argument('--cpu', help='Use cpu', action='store_true')
  parser.add_argument('-m','--model_file', help='File containing trained model to evaluate.',
                      default='seq_arc-4x400_lay-mean_squared_error_opt-adam_act-relu_epo-10_hmean-149p612_hstd-60p171.h5')
  args = parser.parse_args()

  t0 = time()
  device = "CPU" if args.cpu else "GPU"
  tf.device('/'+device+':0')

  model = keras.models.load_model(args.model_file)
  mh_std_train = float(args.model_file.split('_hstd-')[1].split('.h5')[0].replace('p','.'))
  mh_mean_train = float(args.model_file.split('_hmean-')[1].split('_')[0].replace('p','.'))

  test_data_path = ''
  if os.getenv('HOSTNAME'):
    test_data_path = '/net/cms29' 
  test_data_path +='/cms29r0/atto/v1/2016/raw_atto/test_raw_atto_TChiHH_HToBB_HToBB_3D_2016.root'

  eval_model(model, args.model_file.replace('.h5',''), test_data_path, mh_mean_train, mh_std_train)

  print('\nProgram took %.0f:%.0f.' % ((time()-t0)/60,(time()-t0)%60))


