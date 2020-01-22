#! /usr/bin/env python3

import math,os,sys,argparse
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import uproot
from time import time
from coffea import hist
from glob import glob
import utils
from termcolor import colored

def make_sigma_fig(model_name, hsigma):
  fig = plt.figure()
  ax = hist.plot1d(hsigma, overlay="method", stack=False)
  gauss, mask = utils.fit_gauss(hsigma.axis('sigma').centers(), hsigma.values()[('DNN',)], True)
  ax.plot(hsigma.axis('sigma').centers()[mask], gauss, color='maroon', linewidth=1, label=r'Fitted function')
  gauss, mask = utils.fit_gauss(hsigma.axis('sigma').centers(), hsigma.values()[('CB',)], True)
  ax.plot(hsigma.axis('sigma').centers()[mask], gauss, color='navy', linewidth=1, label=r'Fitted function')
  filename = 'sigma_'+model_name+'.pdf'
  fig.savefig(filename)
  print(colored('imgcat '+filename,'green'))
  return

def make_mhiggs_fig(model_name, y_test, y_pred, y_cb):
  fig = plt.figure()
  hmh = hist.Hist("Events",
                      hist.Cat("method", "Reco method"),
                      hist.Bin("mhiggs", "Higgs mass", 30, 0, 300))
  hmh.fill(method="Generated", mhiggs=y_test)
  hmh.fill(method="DNN", mhiggs=y_pred)
  hmh.fill(method="CB", mhiggs=y_cb)
  ax = hist.plot1d(hmh, overlay="method", stack=False)
  filename = 'mh_'+model_name+'.pdf'
  fig.savefig(filename)
  print(colored('imgcat '+filename,'green'))
  return 

def eval_dnn(model, model_name, path, mh_mean_train, mh_std_train, do_log_transform, do_figs=True):
  print('Reading test data: '+colored(path,'yellow'))

  x_test, y_test, mh_mean_test, mh_std_test = utils.get_data(path, do_log_transform)
  y_pred = model.predict(x_test).flatten()

  if do_log_transform:
    # @hack adding 1e-5 because the exp(log()) imprecision makes values jump to neighbor bin
    y_test = np.exp(y_test*mh_std_test + mh_mean_test) +1e-5 
    y_pred = np.exp(y_pred*mh_std_train + mh_mean_train) +1e-5
  else:
    y_test = y_test*mh_std_test + mh_mean_test
    y_pred = y_pred*mh_std_train + mh_mean_train

  sigma_dnn = y_test - y_pred

  # Read some extra branches for evaluation studies
  branches = ['hig_am','nbacc','mchi','mlsp']
  tree = uproot.tree.lazyarrays(path=path, treepath='tree', branches=branches, namedecode='utf-8')
  y_cb = np.asarray(tree['hig_am'])
  sigma_cb = y_test - y_cb

  hsigma = hist.Hist("Events",
                      hist.Cat("method", "Reco method"),
                      hist.Bin("sigma", "True - Predicted", 100, -200, 200))
  hsigma.fill(method="DNN", sigma=sigma_dnn)
  hsigma.fill(method="CB", sigma=sigma_cb)
  # print(hsigma.values())
  print('\nDNN mean = %.2f, std = %.2f' % (np.mean(sigma_dnn), np.std(sigma_dnn)))
  print('CB mean = %.2f, std = %.2f' % (np.mean(sigma_cb), np.std(sigma_cb)))

  if do_figs:
    make_sigma_fig(model_name, hsigma)
    make_mhiggs_fig(model_name, y_test, y_pred, y_cb)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate model performance.')
  parser.add_argument('--cpu', help='Use cpu', action='store_true')
  parser.add_argument('-m','--model_file', help='File containing trained model to evaluate.',
                      default='seq_arc-4x400_lay-mean_squared_error_opt-adam_act-relu_epo-10_hmean-149p612_hstd-60p171.h5')
  args = parser.parse_args()

  t0 = time()
  model = keras.models.load_model(args.model_file)
  mh_std_train = float(args.model_file.split('_hstd-')[1].split('.h5')[0].replace('p','.'))
  mh_mean_train = float(args.model_file.split('_hmean-')[1].split('_')[0].replace('p','.'))

  test_data_path = ''
  if os.getenv('HOSTNAME'):
    test_data_path = '/net/cms29' 
  test_data_path +='/cms29r0/atto/v1/2016/raw_atto/test_raw_atto_TChiHH_HToBB_HToBB_3D_2016.root'

  do_log_transform = ('_log' in args.model_file)
  eval_dnn(model, args.model_file.replace('.h5',''), test_data_path, mh_mean_train, mh_std_train, do_log_transform)

  print('\nProgram took %.0fm %.0fs.' % ((time()-t0)/60,(time()-t0)%60))


