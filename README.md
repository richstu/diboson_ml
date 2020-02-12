# diboson_ml

~~~~bash
git clone git@github.com:richstu/diboson_ml.git
source set_env.sh
~~~~

## Prepare the data 

Done in the nano2pico repository, see README there.

## Training the NN
So far just a simple feed-forward NN, use:

~~~~bash
./train_dnn.py
~~~~

See `-h` for a list of all the options.

## Saving output

Intended to run on merged pico files, preferrably some sort of skim since there is no point in evaluating the DNN for events that will not be looked at. Requires the model saved as .h5 file, as done in `train_dnn.py`. 

~~~~bash
./save_dnn_output.py -i /net/cms29/cms29r0/pico/NanoAODv5/higgsino_eldorado/2016/SMS-TChiHH_2D/higfeats_higloose/ \
                     -m models/MLP5x200_mean_absolute_error_adam_elu_e30_hmean-146p020_hstd-59p933.h5
~~~~

The result is that for each file in the input folder, an output file is created saving the dnn output (just one float per event). The `pico` and `dnnout` file can then be zipped together using `nano2pico/run/update_pico.exe`.

## Evaluating performance on a test sample

More plotting using the pre-trained model to be added to:

~~~~bash
./eval_dnn.py -m -m default_arc-4x400_lay-mean_squared_error_opt-adam_act-relu_epo-400_hmean-149p612_hstd-60p171.h5
~~~~
