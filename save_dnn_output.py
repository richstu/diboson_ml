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
from termcolor import colored, cprint

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run inference for all backgrounds. Intended to run on higfeats ntuples.')
  parser.add_argument('-i','--indir', help='Input folder containing higfeats files.',
                      default='/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/mc/higfeats_preselect/')
  # parser.add_argument('--in_pico', help='Input folder containing pico files.')
  args = parser.parse_args()

  model_names = [
    'nobkg_MLP-5x200_mean_absolute_error_adam_elu_e60',
    'minjj_MLP-5x200_mean_absolute_error_adam_elu_e50',
  ]

  t0 = time()
  models = []
  for name in model_names:
    models.append(keras.models.load_model(name+'.h5'))
  
  do_log = False

  indir = args.indir
  if (indir[-1]!='/'): indir = indir + '/'
  in_file_paths = glob(os.path.join(indir,'*.root'))
  print('Found {} input files.\n'.format(len(in_file_paths)))

  outdir = indir.replace('higfeats_','dnnout_')
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  # in_pico_paths = [args['in_pico']+x.split('/')[-1].replace('higfeats_','') for x in in_file_paths]

  for i in range(len(in_file_paths)):
    if os.path.exists(in_file_paths[i].replace('higfeats_','dnnout_')):
      continue
    print('Processing', in_file_paths[i])
    y_pred = {}
    branches = {}
    for j,imod in enumerate(models):
      x_test, __, __ = utils.get_data(model_name=model_names[j], path=in_file_paths[i], extra_path='', 
        do_log=do_log, training=False, nent=-1)
      label = model_names[j].split('_')[0]
      branches['hig_am_dnn_'+label] = "float32";
      y_pred['hig_am_dnn_'+label] = imod.predict(x_test).flatten()

    with uproot.recreate(in_file_paths[i].replace('higfeats_','dnnout_')) as f:
      f["tree"] = uproot.newtree(branches)
      f["tree"].extend(y_pred)

  print('\nProgram took %.0fm %.0fs.' % ((time()-t0)/60,(time()-t0)%60))


