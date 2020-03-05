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
from scipy.optimize import leastsq
import data_utils

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
  # print("Fit results: coeff = %.2f, mean = %.2f, sigma = %.2f, offset = %.2f" % tuple(fit_result))
  return fit_result,fitfunc(fit_result, xi2), mask

def plot_resolution(tag, y_pred, y_true, y_ref=None):
  sigma_dnn = y_pred - y_true
  if y_ref is not None: sigma_cb = y_ref - y_true

  xmin, xmax, xlabel = -100,350, "True - Predicted"
  if y_true.sum()==0:
    xmin, xmax, xlabel = 50,250, "Higgs mass [GeV]"

  hsigma = hist.Hist("Events",
                      hist.Cat("method", "Reco method"),
                      hist.Bin("sigma", xlabel, 100, xmin, xmax))
  hsigma.fill(method="DNN", sigma=sigma_dnn)
  if y_ref is not None:
    hsigma.fill(method="CB", sigma=sigma_cb)

  fig = plt.figure()
  ax = hist.plot1d(hsigma, overlay="method", stack=False)
  fit_result, gauss, mask = fit_gauss(hsigma.axis('sigma').centers(), hsigma.values()[('DNN',)])
  print('DNN mean = %.2f, std = %.2f' % tuple(fit_result[1:3]))
  
  ax.plot(hsigma.axis('sigma').centers()[mask], gauss, color='maroon', linewidth=1, label=r'Fitted function')
  if y_ref is not None:
    fit_result, gauss, mask = fit_gauss(hsigma.axis('sigma').centers(), hsigma.values()[('CB',)])
    print('Ref. mean = %.2f,x std = %.2f' % tuple(fit_result[1:3]))
    ax.plot(hsigma.axis('sigma').centers()[mask], gauss, color='navy', linewidth=1, label=r'Fitted function')
  filename = 'sigma_'+tag+'.pdf'
  fig.savefig(filename)
  cprint('imgcat '+filename,'green')
  return

def plot_mhiggs(tag, y1, label1, y2, label2, title=''):
  fig = plt.figure()
  hmh = hist.Hist("Events",
                      hist.Cat("process", title),
                      hist.Bin("mhiggs", "Higgs mass [GeV]", 60, 0,300))
  hmh.fill(process=label1, mhiggs=y1)
  hmh.fill(process=label2, mhiggs=y2)
  ax = hist.plot1d(hmh, overlay="process", stack=False)
  filename = tag+'.pdf'
  fig.savefig(filename)
  cprint('imgcat '+filename,'green')
  return 

def find_max_eff(tag, y_pred, y_true, mass_window_width=40): 
  sigma_dnn = y_pred - y_true

  nevents = sigma_dnn.size
  xmin, xmax, bin_width = (-100, 100, 1) if y_true.sum()!=0 else (50,250,1)
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
  print('--> Max. eff.: {:>10s}, sig. eff = {:.0f}%, peak pos = {:.0f} (width = {:.0f})'.format(tag, sig_eff,mass_window_pos,mass_window_width))
  return sig_eff, mass_window_pos

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run inference for all backgrounds. Intended to run on higfeats ntuples.')
  parser.add_argument('--real_higgs', help='Use signal for actual analysis with mH set to 125 GeV.', action='store_true')
  parser.add_argument('--model_name', help='Model name.', default='MLP-5x200_mean_absolute_error_adam_elu_e40')
  args = parser.parse_args()

  model_name = args.model_name

  path_sig, path_bkg = '',''
  if args.real_higgs:
    path_sig = '/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/SMS-TChiHH_2D/higfeats_preselect_nj4/higfeats_merged_pico_preselect_higloose_met150_SMS-TChiHH_mChi-400_mLSP-0_higmc_higloose_nfiles_1.root'
    path_bkg = '/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/mc/higfeats_preselect_nj4//higfeats_merged_pico_preselect_higloose_met150_TTJets_SingleLeptFromT_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_higmc_higloose_nfiles_28.root'
  else: 
    # path_sig = '/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/dnn_mc/TChiHH/higfeats_preselect//higfeats_raw_pico_SMS-TChiHH_HToBB_HToBB_3D_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISummer16NanoAODv5__PUMoriond17_Nano1June2019_102X_mcRun2_asymptotic_v7_test.root'
    path_sig = '/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/dnn_mc/TChiHH/higfeats_preselect//higfeats_raw_pico_preselect_SMS-TChiHH_HToBB_HToBB_3D_TuneCUETP8M1_13TeV-madgraphMLM-pythia8__RunIISummer16NanoAODv5__PUMoriond17_Nano1June2019_102X_mcRun2_asymptotic_v7_test.root'
    path_bkg = ''

  print('Evaluating model performance for: '+colored(model_name,'blue'))
  model = keras.models.load_model(model_name+'.h5')

  # Reading features data for inference
  log_transform = True if "_log" in model_name else False
  if args.real_higgs:
    # signal
    x_data_sig, __ = data_utils.get_data(model_name=model_name, path_sig=path_sig, path_bkg='', do_log=log_transform)
    y_true_sig = np.zeros_like(len(x_data_sig))#125*np.ones_like(len(x_data_sig))
    # background
    x_data_bkg, __ = data_utils.get_data(model_name=model_name, path_sig=path_bkg, path_bkg='', do_log=log_transform)
  else:
    x_data_sig, y_true_sig = data_utils.get_data(model_name=model_name, path_sig=path_sig, path_bkg='', do_log=log_transform)

  
  # Read some extra branches for evaluation studies
  branches = ['hig_cand_am']
  tree_sig = uproot.tree.lazyarrays(path=path_sig, treepath='tree', branches=branches, namedecode='utf-8')
  y_ref_sig = np.asarray(tree_sig['hig_cand_am'])
  if args.real_higgs:
    tree_bkg = uproot.tree.lazyarrays(path=path_bkg, treepath='tree', branches=branches, namedecode='utf-8')
    y_ref_bkg = np.asarray(tree_bkg['hig_cand_am'])

  # Get predicted values
  cprint('\nPerformance plots:')#,'red', attrs=['bold'])
  y_pred_sig = model.predict(x_data_sig).flatten()
  if args.real_higgs: 
    y_pred_bkg = model.predict(x_data_bkg).flatten()

  plot_resolution('sig_'+model_name, y_pred_sig, y_true_sig, y_ref_sig)
  find_max_eff('dnn_'+model_name, y_pred_sig, y_true_sig, mass_window_width=40)
  find_max_eff('ref_'+model_name, y_ref_sig, y_true_sig, mass_window_width=40)
  if args.real_higgs:
    plot_mhiggs('dnn_'+model_name, y_pred_sig, "Signal", y_pred_bkg, "Background", "DNN")
    plot_mhiggs('ref_'+model_name, y_ref_sig, "Signal", y_ref_bkg, "Background", "Reference")
  else:
    plot_mhiggs('mh_'+model_name, y_true_sig, "True value", y_pred_sig, "DNN prediction", "")

  

