#!/bin/bash
# Copyright      2019 Yi Liu
# Apache 2.0
#
# This script will show some backends for 
# The x-vectors of the following datasets should be extracted before running this script.
#   1. SRE04-10 (For source domain PLDA training): sre_04_10_combined
#   2. SRE18-eval (For target domain adaptation): sre18_eval
#   3. SRE18-unlabel (For score normalization): sre18_unlabel
#   4. SRE18-dev-enroll and SRE18-dev-test (As the development set): sre18_dev_enroll, sre18_dev_test
#   5. SRE19-eval-enroll and SRE19-eval-test (As the evaluation set): sre19_eval_enroll, sre19_eval_test
# 
# Basically, there are 2 domains in NIST SRE 19, i.e., the previous-SRE domain (OOD) and CMN domain (ID).
# Only G-PLDA is used for simplicity.
# 
# See README.md for the backend details.
#

. ./cmd.sh
. ./path.sh
set -e

fea_nj=40

root=/home/liuyi/sre19
export data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc

nnetdir=$exp/xvector_nnet_1a

stage=0

lda_dim=200

if [ $stage -le 0 ]; then
  # The mean of each domain should be estimated first.
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp \
    $nnetdir/xvectors_sre_04_10_combined/mean.vec || exit 1;

  $train_cmd $nnetdir/xvectors_sre18_unlabel/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_sre18_unlabel/xvector.scp \
    $nnetdir/xvectors_sre18_unlabel/mean.vec || exit 1;

  exit 1
fi

if [ $stage -le 1 ]; then
  # S1: Use out-of-domain data to train the LDA and PLDA models.
  # Steps: mean norm + LDA + PLDA

  # LDA
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark:- |" \
    ark:$data/sre_04_10_combined/utt2spk $nnetdir/xvectors_sre_04_10_combined/transform.mat || exit 1;

  # PLDA
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre_04_10_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_04_10_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}.txt || exit 1;

  # Scoring
  $train_cmd $nnetdir/xvector_scores/log/sre18_dev.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}.txt - |" \
    "ark:ivector-mean ark:$data/sre18_dev_enroll/spk2utt scp:$nnetdir/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre_04_10_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_04_10_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre18_dev_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre18_dev_plda || exit 1;

  $train_cmd $nnetdir/xvector_scores/log/sre19_eval.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}.txt - |" \
    "ark:ivector-mean ark:$data/sre19_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre_04_10_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_04_10_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre19_eval_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre19_eval_plda || exit 1;

  python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores/sre18_dev_plda $nnetdir/xvector_scores/sre18_dev_plda.cmn
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre18_dev_plda.cmn $nnetdir/xvector_scores/sre18_dev_plda.vast > $nnetdir/xvector_scores/sre18_dev_plda.all

  python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores/sre19_eval_plda $nnetdir/xvector_scores/sre19_eval_plda.cts
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre19_eval_plda.cts > $nnetdir/xvector_scores/sre19_eval_plda.all

  source activate sre18
  cd ./sre18_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre18_dev_plda.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
  cd -

  cd ./cts_challenge_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre19_eval_plda.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
  cd -
  source deactivate

  exit 1
fi

if [ $stage -le 2 ]; then
  # S2: Use the in-domain data to train the LDA and PLDA models
  # Steps: mean norm + LDA + PLDA

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark:- |" \
    ark:$data/sre18_eval/utt2spk $nnetdir/xvectors_sre18_eval/transform.mat || exit 1;

  # PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre18_eval/spk2utt \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt || exit 1;

  # Scoring
  $train_cmd $nnetdir/xvector_scores/log/sre18_dev.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt - |" \
    "ark:ivector-mean ark:$data/sre18_dev_enroll/spk2utt scp:$nnetdir/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre18_dev_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre18_dev_plda || exit 1;

  $train_cmd $nnetdir/xvector_scores/log/sre19_eval.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt - |" \
    "ark:ivector-mean ark:$data/sre19_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre19_eval_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre19_eval_plda || exit 1;

  python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores/sre18_dev_plda $nnetdir/xvector_scores/sre18_dev_plda.cmn
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre18_dev_plda.cmn $nnetdir/xvector_scores/sre18_dev_plda.vast > $nnetdir/xvector_scores/sre18_dev_plda.all

  python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores/sre19_eval_plda $nnetdir/xvector_scores/sre19_eval_plda.cts
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre19_eval_plda.cts > $nnetdir/xvector_scores/sre19_eval_plda.all

  source activate sre18
  cd ./sre18_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre18_dev_plda.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
  cd -

  cd ./cts_challenge_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre19_eval_plda.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
  cd -
  source deactivate

  exit 1
fi

if [ $stage -le 3 ]; then
  # S3: Use the mixed (ooD + inD) to train the LDA and PLDA models
  # Steps: mean norm + LDA + PLDA
  
  # Mix the ooD and inD data
  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval/xvector_mean.ark,$nnetdir/xvectors_sre18_eval/xvector_mean.scp
  cat $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  # cat $data/sre_04_10_combined/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  cat $data/sre_04_10/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
    ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt \
    "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}.txt || exit 1;

  # Scoring
  $train_cmd $nnetdir/xvector_scores/log/sre18_dev.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}.txt - |" \
    "ark:ivector-mean ark:$data/sre18_dev_enroll/spk2utt scp:$nnetdir/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre18_dev_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre18_dev_plda || exit 1;

  $train_cmd $nnetdir/xvector_scores/log/sre19_eval.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}.txt - |" \
    "ark:ivector-mean ark:$data/sre19_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre19_eval_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre19_eval_plda || exit 1;

  python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores/sre18_dev_plda $nnetdir/xvector_scores/sre18_dev_plda.cmn
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre18_dev_plda.cmn $nnetdir/xvector_scores/sre18_dev_plda.vast > $nnetdir/xvector_scores/sre18_dev_plda.all

  python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores/sre19_eval_plda $nnetdir/xvector_scores/sre19_eval_plda.cts
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre19_eval_plda.cts > $nnetdir/xvector_scores/sre19_eval_plda.all

  source activate sre18
  cd ./sre18_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre18_dev_plda.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
  cd -

  cd ./cts_challenge_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre19_eval_plda.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
  cd -
  source deactivate

  exit 1
fi

if [ $stage -le 4 ]; then
  # S4: Mixed the ooD and inD data and adjust the within and between covariances of the PLDA model.
  # Steps: mean norm + LDA + PLDA (adapted)

  # Mix the data
  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval/xvector_mean.ark,$nnetdir/xvectors_sre18_eval/xvector_mean.scp
  cat $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  # cat $data/sre_04_10_combined/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  cat $data/sre_04_10/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
    ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt \
    "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}.txt || exit 1;

  # Adjust the PLDA (using the mixed data). This just adjust the covariances of the model (towards larger covariances).
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}.txt \
      "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}_adapt || exit 1;

  # Scoring
  $train_cmd $nnetdir/xvector_scores/log/sre18_dev.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre18_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}_adapt - |" \
    "ark:ivector-mean ark:$data/sre18_dev_enroll/spk2utt scp:$nnetdir/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre18_dev_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre18_dev_plda || exit 1;

  # Scoring
  $train_cmd $nnetdir/xvector_scores/log/sre19_eval.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre19_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}_adapt - |" \
    "ark:ivector-mean ark:$data/sre19_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre19_eval_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre19_eval_plda || exit 1;

  python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores/sre18_dev_plda $nnetdir/xvector_scores/sre18_dev_plda.cmn
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre18_dev_plda.cmn $nnetdir/xvector_scores/sre18_dev_plda.vast > $nnetdir/xvector_scores/sre18_dev_plda.all

  python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores/sre19_eval_plda $nnetdir/xvector_scores/sre19_eval_plda.cts
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre19_eval_plda.cts > $nnetdir/xvector_scores/sre19_eval_plda.all

  source activate sre18
  cd ./sre18_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre18_dev_plda.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
  cd -

  cd ./cts_challenge_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre19_eval_plda.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
  cd -
  source deactivate

  exit 1
fi

if [ $stage -le 5 ]; then
  # S5: As the same as S4, while an additional AS-Norm is performed
  # Steps: mean norm + LDA + PLDA (adapted) + AS-Norm
  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval/xvector_mean.ark,$nnetdir/xvectors_sre18_eval/xvector_mean.scp
  cat $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  # cat $data/sre_04_10_combined/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  cat $data/sre_04_10/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
    ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt \
    "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}.txt || exit 1;

  # Adjust the PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}.txt \
      "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}_adapt || exit 1;

  # AS-Norm
  backend_utils/as_norm.sh $nnetdir/xvectors_sre18_unlabel/mean.vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat $nnetdir/xvectors_sre18_eval_combined_with_sre/plda_lda${lda_dim}_adapt $nnetdir/xvectors_sre18_unlabel $nnetdir/xvectors_sre18_dev_enroll $nnetdir/xvectors_sre18_dev_test $nnetdir/xvectors_sre19_eval_enroll $nnetdir/xvectors_sre19_eval_test $nnetdir/xvector_scores

  python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores/asnorm_score_sre18_norm $nnetdir/xvector_scores/asnorm_score_sre18_norm.cmn
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/asnorm_score_sre18_norm.vast
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/asnorm_score_sre18_norm.cmn $nnetdir/xvector_scores/asnorm_score_sre18_norm.vast > $nnetdir/xvector_scores/asnorm_score_sre18_norm.all

  python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores/asnorm_score_sre19_norm $nnetdir/xvector_scores/asnorm_score_sre19_norm.cts
  cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/asnorm_score_sre19_norm.cts > $nnetdir/xvector_scores/asnorm_score_sre19_norm.all

  source activate sre18
  cd ./sre18_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/asnorm_score_sre18_norm.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
  cd -

  cd ./cts_challenge_scoring_software
  python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/asnorm_score_sre19_norm.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
  cd -
  source deactivate

  exit 1
fi

if [ $stage -le 6 ]; then
  # S6: Use supervised PLDA adaptation.
  # Steps: mean norm + LDA + whitening + PLDA (interpolated)
  # Ref:
  #   Garcia-Romero D, Mccree A. Supervised domain adaptation for I-vector based speaker recognition, 2014[C].May.

  # Mix the data
  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval/xvector_mean.ark,$nnetdir/xvectors_sre18_eval/xvector_mean.scp
  cat $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  cat $data/sre_04_10_combined/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
    ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # Whitening
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/whiten.log \
    est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true \
      "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- |" \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat

  # ooD PLDA
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre_04_10_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}.txt || exit 1;

  # inD PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre18_eval/spk2utt \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt || exit 1;

  # Convert the PLDA model to text format for convenience.
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}
    
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast

  for interpolate in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ; do
  (
    # PLDA interpolation
    python backend_utils/plda_interpolate.py $interpolate $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim} $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim} $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}

    # Convert the PLDA back to Kaldi format
    python backend_utils/txt_to_plda.py $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/mean.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/transform.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/psi.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt
    
    # Scoring
    $train_cmd $nnetdir/xvector_scores/log/sre18_dev_${interpolate}.log \
      ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:$nnetdir/xvectors_sre18_dev_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt - |" \
      "ark:ivector-mean ark:$data/sre18_dev_enroll/spk2utt scp:$nnetdir/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$data/sre18_dev_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre18_dev_plda_${interpolate} || exit 1;
  
    # Scoring
    $train_cmd $nnetdir/xvector_scores/log/sre19_eval_${interpolate}.log \
      ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:$nnetdir/xvectors_sre19_eval_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt - |" \
      "ark:ivector-mean ark:$data/sre19_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$data/sre19_eval_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre19_eval_plda_${interpolate} || exit 1;
  
    python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores/sre18_dev_plda_${interpolate} $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.cmn
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.cmn $nnetdir/xvector_scores/sre18_dev_plda.vast > $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.all
  
    python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores/sre19_eval_plda_${interpolate} $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.cts
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.cts > $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.all
  
    source activate sre18
    cd ./sre18_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
    cd -
  
    cd ./cts_challenge_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
    cd -
    source deactivate
  ) > result_${interpolate}.txt &
  done 

  wait 
  exit 1
fi

if [ $stage -le 7 ]; then
  # S7: Same as S6 while an additional AS-Norm is performed.
  # Steps: mean norm + LDA + whitening + PLDA (interpolated) + AS-Norm

  # Mix the data
  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval/xvector_mean.ark,$nnetdir/xvectors_sre18_eval/xvector_mean.scp
  cat $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  cat $data/sre_04_10_combined/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
    ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # Whitening
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/whiten.log \
    est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true \
      "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- |" \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat

  # ooD PLDA
  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre_04_10_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}.txt || exit 1;

  # inD PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre18_eval/spk2utt \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt || exit 1;

  # Convert the PLDA model to text format for convenience.
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim}
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}
    
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast

  for interpolate in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0 ; do
  (
    # PLDA interpolation
    python backend_utils/plda_interpolate.py $interpolate $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim} $nnetdir/xvectors_sre_04_10_combined/plda_lda${lda_dim} $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}

    # Convert the PLDA back to Kaldi format
    python backend_utils/txt_to_plda.py $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/mean.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/transform.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/psi.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt
    
    # AS-Norm
    backend_utils/as_norm_with_whiten.sh $nnetdir/xvectors_sre18_unlabel/mean.vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_unlabel $nnetdir/xvectors_sre18_dev_enroll $nnetdir/xvectors_sre18_dev_test $nnetdir/xvectors_sre19_eval_enroll $nnetdir/xvectors_sre19_eval_test $nnetdir/xvector_scores_${interpolate}

    python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.cmn
    tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.vast
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.cmn $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.vast > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.all

    python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.cts
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.cts > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.all

    source activate sre18
    cd ./sre18_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
    cd -

    cd ./cts_challenge_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
    cd -
    source deactivate
  ) > result_${interpolate}.txt &
  done 

  wait 
  exit 1
fi

if [ $stage -le 8 ]; then
  # S8: Before the backend, apply the CORAL transform to the out-of-domain data.
  # Steps: mean norm + CORAL (to ooD data) + LDA + whitening + PLDA (interpolated)
  # Ref: 
  #   Alam M J, Bhattacharya G, Kenny P. Speaker Verification in Mismatched Conditions with Frustratingly Easy Domain Adaptation, 2018[C]. 
  # I also try FDA from LIA, but CORAL performs better in my case.

  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval/xvector_mean.ark,$nnetdir/xvectors_sre18_eval/xvector_mean.scp

  # Perform CORAL
  python backend_utils/coral_fit_and_transform.py $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp $nnetdir/xvectors_sre_04_10_combined_transformed
  # Or you can optionally perform FDA as follow:
  # python backend_utils/fda_fit_and_transform.py $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp $nnetdir/xvectors_sre_04_10_combined_transformed

  # Some ugly transform
  $train_cmd $nnetdir/xvectors_sre_04_10_combined_transformed/log/copy_vector.log \
    copy-vector ark:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector_transformed.ark ark,scp:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.ark,$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp
  rm $nnetdir/xvectors_sre_04_10_combined_transformed/xvector_transformed.ark

  # Mix the data
  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  cat $nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  cat $data/sre_04_10_combined/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
      scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
      ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # Whitening
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/whiten.log \
    est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true \
      "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- |" \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat

  # ooD PLDA
  $train_cmd $nnetdir/xvectors_sre_04_10_combined_transformed/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre_04_10_combined/spk2utt \
    "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}.txt || exit 1;

  # inD PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre18_eval/spk2utt \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt || exit 1;

  # Convert the PLDA model to text format for convenience.
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}
    
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast

  for interpolate in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0 ; do
  (
    # PLDA interpolation
    python backend_utils/plda_interpolate.py $interpolate $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim} $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim} $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}

    # Convert the PLDA back to Kaldi format
    python backend_utils/txt_to_plda.py $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/mean.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/transform.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/psi.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt
    
    # Scoring
    $train_cmd $nnetdir/xvector_scores/log/sre18_dev_${interpolate}.log \
      ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:$nnetdir/xvectors_sre18_dev_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt - |" \
      "ark:ivector-mean ark:$data/sre18_dev_enroll/spk2utt scp:$nnetdir/xvectors_sre18_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_dev_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$data/sre18_dev_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre18_dev_plda_${interpolate} || exit 1;
  
    # Scoring
    $train_cmd $nnetdir/xvector_scores/log/sre19_eval_${interpolate}.log \
      ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:$nnetdir/xvectors_sre19_eval_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt - |" \
      "ark:ivector-mean ark:$data/sre19_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre19_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre19_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$data/sre19_eval_test/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre19_eval_plda_${interpolate} || exit 1;
  
    python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores/sre18_dev_plda_${interpolate} $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.cmn
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.cmn $nnetdir/xvector_scores/sre18_dev_plda.vast > $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.all
  
    python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores/sre19_eval_plda_${interpolate} $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.cts
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.cts > $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.all
  
    source activate sre18
    cd ./sre18_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre18_dev_plda_${interpolate}.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
    cd -
  
    cd ./cts_challenge_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores/sre19_eval_plda_${interpolate}.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
    cd -
    source deactivate
  ) > result_${interpolate}.txt &
  done 

  wait 
  exit 1
fi

if [ $stage -le 9 ]; then
  # S9: Same as S8, while AS-Norm is performed.  
  # Steps: mean norm + CORAL (to ooD data) + LDA + whitening + PLDA (interpolated) + AS-Norm

  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval/xvector_mean.ark,$nnetdir/xvectors_sre18_eval/xvector_mean.scp

  # Perform CORAL
  python backend_utils/coral_fit_and_transform.py $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp $nnetdir/xvectors_sre_04_10_combined_transformed
  $train_cmd $nnetdir/xvectors_sre_04_10_combined_transformed/log/copy_vector.log \
    copy-vector ark:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector_transformed.ark ark,scp:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.ark,$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp
  rm $nnetdir/xvectors_sre_04_10_combined_transformed/xvector_transformed.ark

  # Mix the data
  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  cat $nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp $nnetdir/xvectors_sre18_eval/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  cat $data/sre_04_10_combined/utt2spk $data/sre18_eval/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
    ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # Whitening
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/whiten.log \
    est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true \
      "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- |" \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat

  # ooD PLDA
  $train_cmd $nnetdir/xvectors_sre_04_10_combined_transformed/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre_04_10_combined/spk2utt \
    "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}.txt || exit 1;

  # inD PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre18_eval/spk2utt \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt || exit 1;

  # Convert the PLDA model to text format for convenience.
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim}
    
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast

  for interpolate in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0 ; do
  (
    # PLDA interpolation
    python backend_utils/plda_interpolate.py $interpolate $nnetdir/xvectors_sre18_eval/plda_lda${lda_dim} $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim} $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}

    # Convert the PLDA back to Kaldi format
    python backend_utils/txt_to_plda.py $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/mean.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/transform.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/psi.txt $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt
    
    # AS-Norm
    backend_utils/as_norm_with_whiten.sh $nnetdir/xvectors_sre18_unlabel/mean.vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat $nnetdir/xvectors_sre18_eval/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_unlabel $nnetdir/xvectors_sre18_dev_enroll $nnetdir/xvectors_sre18_dev_test $nnetdir/xvectors_sre19_eval_enroll $nnetdir/xvectors_sre19_eval_test $nnetdir/xvector_scores_${interpolate}

    python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.cmn
    tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.vast
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.cmn $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.vast > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.all

    python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.cts
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.cts > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.all

    source activate sre18
    cd ./sre18_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
    cd -

    cd ./cts_challenge_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
    cd -
    source deactivate
  ) > result_${interpolate}.txt &
  done 

  wait 
  exit 1
fi

if [ $stage -le 10 ]; then
  # S10: Add augmented in-domain data which I think will make the in-domain model more robust.
  # Steps: mean norm + CORAL (to ooD data) + LDA + whitening + PLDA (interpolated) + AS-Norm

  # Make the augmented in-domain training data
  mkdir -p $nnetdir/xvectors_sre18_eval_combined
  cat $nnetdir/xvectors_sre18_eval/xvector.scp $nnetdir/xvectors_sre18_eval_aug_15000/xvector.scp > $nnetdir/xvectors_sre18_eval_combined/xvector.scp
  utils/combine_data.sh $data/sre18_eval_combined $data/sre18_eval $data/sre18_eval_aug_15000

  $train_cmd $nnetdir/xvectors_sre_04_10_combined/log/compute_mean.log \
    ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_04_10_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre_04_10_combined/xvector_mean.ark,$nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp
  $train_cmd $nnetdir/xvectors_sre18_eval_combined/log/compute_mean.log \
    ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval_combined/xvector.scp ark,scp:$nnetdir/xvectors_sre18_eval_combined/xvector_mean.ark,$nnetdir/xvectors_sre18_eval_combined/xvector_mean.scp

  # Perform CORAL
  python backend_utils/coral_fit_and_transform.py $nnetdir/xvectors_sre_04_10_combined/xvector_mean.scp $nnetdir/xvectors_sre18_eval_combined/xvector_mean.scp $nnetdir/xvectors_sre_04_10_combined_transformed
  $train_cmd $nnetdir/xvectors_sre_04_10_combined_transformed/log/copy_vector.log \
    copy-vector ark:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector_transformed.ark ark,scp:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.ark,$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp
  rm $nnetdir/xvectors_sre_04_10_combined_transformed/xvector_transformed.ark

  mkdir -p $nnetdir/xvectors_sre18_eval_combined_with_sre
  cat $nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp $nnetdir/xvectors_sre18_eval_combined/xvector_mean.scp > $nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp
  cat $data/sre_04_10_combined/utt2spk $data/sre18_eval_combined/utt2spk | sort > $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk
  utils/utt2spk_to_spk2utt.pl $nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk > $nnetdir/xvectors_sre18_eval_combined_with_sre/spk2utt

  # LDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp \
    ark:$nnetdir/xvectors_sre18_eval_combined_with_sre/utt2spk $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat || exit 1;

  # Whitening
  $train_cmd $nnetdir/xvectors_sre18_eval_combined_with_sre/log/whiten.log \
    est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true \
      "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre18_eval_combined_with_sre/xvector_mean_combined_with_sre.scp ark:- |" \
      $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat

  # ooD PLDA
  $train_cmd $nnetdir/xvectors_sre_04_10_combined_transformed/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre_04_10_combined/spk2utt \
    "ark:transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat scp:$nnetdir/xvectors_sre_04_10_combined_transformed/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}.txt || exit 1;

  # inD PLDA
  $train_cmd $nnetdir/xvectors_sre18_eval_combined/log/plda_lda${lda_dim}.log \
    ivector-compute-plda --binary=false ark:$data/sre18_eval_combined/spk2utt \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre18_unlabel/mean.vec scp:$nnetdir/xvectors_sre18_eval_combined/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat ark:- ark:- | transform-vec $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre18_eval_combined/plda_lda${lda_dim}.txt || exit 1;

  # Convert the PLDA model to text format for convenience.
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim}
  python backend_utils/plda_to_txt.py $nnetdir/xvectors_sre18_eval_combined/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_eval_combined/plda_lda${lda_dim}
    
  tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores/sre18_dev_plda.vast

  for interpolate in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0 ; do
  (
    # PLDA interpolation
    python backend_utils/plda_interpolate.py $interpolate $nnetdir/xvectors_sre18_eval_combined/plda_lda${lda_dim} $nnetdir/xvectors_sre_04_10_combined_transformed/plda_lda${lda_dim} $nnetdir/xvectors_sre18_eval_combined/plda_interpolate_${interpolate}

    # Convert the PLDA back to Kaldi format
    python backend_utils/txt_to_plda.py $nnetdir/xvectors_sre18_eval_combined/plda_interpolate_${interpolate}/mean.txt $nnetdir/xvectors_sre18_eval_combined/plda_interpolate_${interpolate}/transform.txt $nnetdir/xvectors_sre18_eval_combined/plda_interpolate_${interpolate}/psi.txt $nnetdir/xvectors_sre18_eval_combined/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt

    # AS-Norm 
    backend_utils/as_norm_with_whiten.sh $nnetdir/xvectors_sre18_unlabel/mean.vec $nnetdir/xvectors_sre18_eval_combined_with_sre/transform.mat $nnetdir/xvectors_sre18_eval_combined_with_sre/whiten.mat $nnetdir/xvectors_sre18_eval_combined/plda_interpolate_${interpolate}/plda_lda${lda_dim}.txt $nnetdir/xvectors_sre18_unlabel $nnetdir/xvectors_sre18_dev_enroll $nnetdir/xvectors_sre18_dev_test $nnetdir/xvectors_sre19_eval_enroll $nnetdir/xvectors_sre19_eval_test $nnetdir/xvector_scores_${interpolate}

    python backend_utils/combine_single_scores.py sph $data/sre18_dev_test/trials $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.cmn
    tail -n +2 docs/sre18_dev_docs/sre18_dev_trials_vast.tsv | awk '{print $0"\t0.0"}' > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.vast
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.cmn $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.vast > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.all

    python backend_utils/combine_single_scores.py sph $data/sre19_eval_test/trials $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.cts
    cat docs/sre18_dev_docs/header.tsv $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.cts > $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.all

    source activate sre18
    cd ./sre18_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre18_norm.all -l ../docs/sre18_dev_docs/sre18_dev_trials.tsv -r ../docs/sre18_dev_docs/sre18_dev_trial_key.tsv
    cd -

    cd ./cts_challenge_scoring_software
    python3 sre18_submission_scorer.py -o $nnetdir/xvector_scores_${interpolate}/asnorm_score_sre19_norm.all -l ../docs/sre19_eval_docs/sre19_cts_challenge_trials.tsv -r ../docs/sre19_eval_docs/sre19_cts_challenge_trial_key.tsv
    cd -
    source deactivate
  ) > result_${interpolate}.txt &
  done 

  wait 
  exit 1
fi

