# diboson_ml

~~~~bash
git clone git@github.com:richstu/diboson_ml.git
source set_env.sh
~~~~

# Prepare the data (nano2pico)

Run make_atto in the nano2pico repository, e.g.:

~~~~bash
./scripts/write_make_atto_cmds.py -i /net/cms29/cms29r0/atto/nano/2016/ -v v2
~~~~

The output folder would then be set to `/net/cms29/cms29r0/atto/v2/2016/raw_atto/`. For easier subsequent processing it's best to split the sample into training and testing samples and merge each of these into a single root file:

~~~~bash
cd /net/cms29/cms29r0/atto/v2/2016/raw_atto/
mkdir test train
# the files starting with the digit 5 are approximately 20%, so set these aside for the test
mv raw_atto_TChiHH_HToBB_HToBB_3D_2016_file5* test/
mv raw_atto_TChiHH_HToBB_HToBB_3D_2016_file* train/
cd -
./scripts/slim_and_merge.py -s txt/slim_rules/all.txt \
                            -o /cms29r0/atto/v2/2016/raw_atto/test_raw_atto_TChiHH_HToBB_HToBB_3D_2016.root \
                            -i /cms29r0/atto/v2/2016/raw_atto/test/*root
./scripts/slim_and_merge.py -s txt/slim_rules/all.txt \
                            -o /cms29r0/atto/v2/2016/raw_atto/train_raw_atto_TChiHH_HToBB_HToBB_3D_2016.root \
                            -i /cms29r0/atto/v2/2016/raw_atto/train/*root
~~~~

# Training the NN
So far just a simple feed-forward NN, use:

~~~~bash
./train_dnn.py
~~~~

See `-h` for a list of all the options.

# Evaluating performance on a test sample

More plotting using the pre-trained model to be added to:

~~~~bash
./eval_dnn.py -m -m default_arc-4x400_lay-mean_squared_error_opt-adam_act-relu_epo-400_hmean-149p612_hstd-60p171.h5
~~~~
