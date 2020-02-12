#! /usr/bin/env python3

import math,os,sys,argparse
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import uproot
from time import time
from coffea import hist
from glob import glob
import eval_utils
from termcolor import colored, cprint

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run inference for all backgrounds. Intended to run on higfeats ntuples.')
  parser.add_argument('-i','--indir', help='Input folder containing higfeats files.',
                      default='/net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/SMS-TChiHH_2D/higfeats_higloose/')
  # parser.add_argument('--in_pico', help='Input folder containing pico files.')
  parser.add_argument('-m','--model_file', help='File containing trained model to evaluate.',
                      default='MLP5x200_mean_absolute_error_adam_elu_e30_log.h5')
  args = parser.parse_args()

  t0 = time()
  model = keras.models.load_model(args.model_file)
  do_log = ('_log' in args.model_file)

  indir = args.indir
  if (indir[-1]!='/'): indir = indir + '/'
  in_file_paths = glob(os.path.join(indir,'*_mLSP-0*.root'))
  print('Found {} input files.\n'.format(len(in_file_paths)))

  outdir = indir.replace('higfeats_','dnnout_')
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  # in_pico_paths = [args['in_pico']+x.split('/')[-1].replace('higfeats_','') for x in in_file_paths]

  for i in range(len(in_file_paths)):
    if os.path.exists(in_file_paths[i].replace('higfeats_','dnnout_')):
      continue
    print('Processing', in_file_paths[i])
    __, y_pred = eval_utils.get_predictions(model, in_file_paths[i], do_log)
    print(y_pred)
    print('Number of entries in y_pred',len(y_pred))

    branches = ['njet', 'mprod','mlsp']
    tree = uproot.tree.lazyarrays(path=in_file_paths[i], treepath='tree', branches=branches, namedecode='utf-8')
    print('Number of entries in tree',len(tree['njet']))

    with uproot.recreate(in_file_paths[i].replace('higfeats_','dnnout_')) as f:
      f["tree"] = uproot.newtree({"hig_am_dnn": "float64", "mprod": "int", "njet": "int"})
      f["tree"].extend({"hig_am_dnn": y_pred, "mprod": tree['mprod'], "njet": tree['njet']})

  print('\nProgram took %.0fm %.0fs.' % ((time()-t0)/60,(time()-t0)%60))


