#! /usr/bin/env python3

import math,os,sys,argparse
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import uproot
from time import time
from coffea import hist
from glob import glob
from termcolor import colored, cprint
from collections import OrderedDict

def fit_gauss(xi, yi, verbose=False):
  fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
  errfunc  = lambda p, x, y: (y - fitfunc(p, x))
  
  init  = [yi.max(), xi[yi.argmax()], 30, 0.01]
  fit_result, fit_cov  = leastsq( errfunc, init, args=(xi, yi))
  if (verbose):
    print("Init values 1: coeff = %.2f, mean = %.2f, sigma = %.2f, offset = %.2f" % tuple(init))
    print("Fit results 1: coeff = %.2f, mean = %.2f, sigma = %.2f, offset = %.2f" % tuple(fit_result))
  
  # redo fit just taking 2*sigma around the peak
  mask = np.logical_and(xi > (fit_result[1]-1.5*abs(fit_result[2])), xi < (fit_result[1]+1.5*abs(fit_result[2])))
  xi2, yi2 = xi[mask], yi[mask]
  fit_result, fit_cov  = leastsq( errfunc, fit_result, args=(xi2, yi2))
  if (verbose):
    print("Fit results 2: coeff = %.2f, mean = %.2f, sigma = %.2f, offset = %.2f" % tuple(fit_result))
  return fitfunc(fit_result, xi2), mask

def plot_resolution(tag, y_test, y_pred, y_cb):
  sigma_dnn = y_test - y_pred
  sigma_cb = y_test - y_cb

  hsigma = hist.Hist("Events",
                      hist.Cat("method", "Reco method"),
                      hist.Bin("sigma", "True - Predicted", 80, -200, 200))
  hsigma.fill(method="DNN", sigma=sigma_dnn)
  hsigma.fill(method="CB", sigma=sigma_cb)
  # print(hsigma.values())
  print('\nDNN mean = %.2f, std = %.2f' % (np.mean(sigma_dnn), np.std(sigma_dnn)))
  print('CB mean = %.2f, std = %.2f' % (np.mean(sigma_cb), np.std(sigma_cb)))

  fig = plt.figure()
  ax = hist.plot1d(hsigma, overlay="method", stack=False)
  gauss, mask = fit_gauss(hsigma.axis('sigma').centers(), hsigma.values()[('DNN',)])
  ax.plot(hsigma.axis('sigma').centers()[mask], gauss, color='maroon', linewidth=1, label=r'Fitted function')
  gauss, mask = fit_gauss(hsigma.axis('sigma').centers(), hsigma.values()[('CB',)])
  ax.plot(hsigma.axis('sigma').centers()[mask], gauss, color='navy', linewidth=1, label=r'Fitted function')
  filename = 'sigma_'+tag+'.pdf'
  fig.savefig(filename)
  cprint('imgcat '+filename,'green')
  return

def plot_mhiggs(tag, y_test, y_pred, y_cb):
  fig = plt.figure()
  hmh = hist.Hist("Events",
                      hist.Cat("method", "Reco method"),
                      hist.Bin("mhiggs", "Higgs mass", 30, 0, 300))
  hmh.fill(method="Generated", mhiggs=y_test)
  hmh.fill(method="DNN", mhiggs=y_pred)
  hmh.fill(method="CB", mhiggs=y_cb)
  ax = hist.plot1d(hmh, overlay="method", stack=False)
  filename = 'mh_'+tag+'.pdf'
  fig.savefig(filename)
  cprint('imgcat '+filename,'green')
  return 

def find_max_eff(tag, y_test, y_pred, mass_window_width=40): 
  sigma_dnn = y_test - y_pred

  nevents = sigma_dnn.size
  xmin, xmax, bin_width = -100, 100, 1
  if (mass_window_width%bin_width>0):
    sys.exit("eval_dnn::find_max_eff() Mass window must be divisible by bin width")
  counts, edges = np.histogram(sigma_dnn, bins=int((xmax-xmin)/bin_width), range=(xmin, xmax))

  # find window that would contain the most events
  isum = counts[0:mass_window_width].sum()
  max_sum = isum
  mass_window_pos = xmin+mass_window_width/2.
  for i in range(len(counts)):
    prev_bin, next_bin = i, i+int(mass_window_width/bin_width)
    if next_bin >= len(counts): 
      break
    isum += counts[next_bin]-counts[prev_bin]
    if isum>max_sum:
      max_sum = isum
      mass_window_pos = xmin + i*bin_width + mass_window_width/2.

  sig_eff = float(max_sum)/nevents*100.
  print('--> Method: {:>10s}, sig. eff = {:.0f}%, peak pos = {:.0f} (width = {:.0f})'.format(tag, sig_eff,mass_window_pos,mass_window_width))
  return sig_eff, mass_window_pos

def get_predictions(model, test_data_path, mh_mean_train, mh_std_train, do_log_transform):
  print('Reading test data: '+colored(test_data_path,'yellow'))
  x_test, y_test, mh_mean_test, mh_std_test = utils.get_data(test_data_path, do_log_transform)
  y_pred = model.predict(x_test).flatten()
  if do_log_transform:
    # @hack adding 1e-5 because the exp(log()) imprecision makes values jump to neighbor bin
    y_test = np.exp(y_test*mh_std_test + mh_mean_test) +1e-5 
    y_pred = np.exp(y_pred*mh_std_train + mh_mean_train) +1e-5
  else:
    y_test = y_test*mh_std_test + mh_mean_test
    y_pred = y_pred*mh_std_train + mh_mean_train
  return y_test, y_pred

def eval_dnn(model, model_name, test_data_path, mh_mean_train, mh_std_train, do_log_transform, do_figs=True):
  y_test, y_pred = get_predictions(model, test_data_path, mh_mean_train, mh_std_train, do_log_transform)
  # Read some extra branches for evaluation studies
  branches = ['hig_am','nbacc','njet', 'mchi','mlsp']
  tree = uproot.tree.lazyarrays(path=test_data_path, treepath='tree', branches=branches, namedecode='utf-8')
  y_cb = np.asarray(tree['hig_am'])

  seln_dict = OrderedDict()
  seln_dict['none'] = None
  seln_dict['nbacc_geq_4'] = np.asarray(tree['nbacc'])>=4
  # seln_dict['njet4'] = (seln_dict['nbacc_geq_4']) & (np.asarray(tree['njet'])==4)

  for iseln,mask in seln_dict.items():
    cprint('\nEvaluating selection: '+iseln,'yellow')
    if mask is not None:
      y_test_good, y_pred_good, y_cb_good = y_test[mask], y_pred[mask], y_cb[mask]
    else:
      y_test_good, y_pred_good, y_cb_good = y_test, y_pred, y_cb

    find_max_eff('DNN__'+iseln, y_test_good, y_pred_good, mass_window_width=40)
    find_max_eff('CB__'+iseln, y_test_good, y_cb_good, mass_window_width=40)

    tag = iseln+'__'+model_name  # for filename
    plot_resolution(tag, y_test_good, y_pred_good, y_cb_good)
    plot_mhiggs(tag, y_test_good, y_pred_good, y_cb_good)

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


