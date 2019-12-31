#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

data_in=$1
data_out=$2

data=$(utils/make_absolute.sh $data_in/../)
rirs_noises=/home/liuyi/data/rirs_noises

if [ ! -d $rirs_noises ]; then
  echo "Please set rirs_noises properly."
  exit 1
fi

if [ ! -d $data/musan_noise ] || [ ! -d $data/musan_speech ] || [ ! -d $data/musan_music ]; then
  echo "Please make musan dataset using the Kaldi script (see sre16/v2/run.sh)."

# Example:  
# local/make_musan_8k.sh $musan $data
# for name in speech noise music; do
#   utils/data/get_utt2dur.sh $data/musan_${name}
#   mv $data/musan_${name}/utt2dur $data/musan_${name}/reco2dur
# done

  exit 1
fi

if [ ! -f $data_in/utt2num_frames ]; then
  feat-to-len scp:$data_in/feats.scp ark,t:$data_in/utt2num_frames
fi

frame_shift=0.01
awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data_in/utt2num_frames > $data_in/reco2dur

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/mediumroom/rir_list")

# Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
# additive noise here.
python3 steps/data/reverberate_data_dir.py \
  "${rvb_opts[@]}" \
  --speech-rvb-probability 1 \
  --pointsource-noise-addition-probability 0 \
  --isotropic-noise-addition-probability 0 \
  --num-replications 1 \
  --source-sampling-rate 8000 \
  $data_in $data_out/reverb
cp $data_in/vad.scp $data_out/reverb/
utils/copy_data_dir.sh --utt-suffix "-reverb" $data_out/reverb $data_out/reverb.new
rm -rf $data_out/reverb
mv $data_out/reverb.new $data_out/reverb

utils/fix_data_dir.sh $data_out/reverb

# Augment with musan_noise
python3 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$data/musan_noise" $data_in $data_out/noise
# Augment with musan_music
python3 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$data/musan_music" $data_in $data_out/music
# Augment with musan_speech
python3 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$data/musan_speech" $data_in $data_out/babble

# Combine reverb, noise, music, and babble into one directory.
utils/combine_data.sh $data_out/combined $data_out/reverb $data_out/noise $data_out/music $data_out/babble 
cp $data_out/combined/* $data_out

