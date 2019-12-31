#!/bin/bash
# 
# Do the AS-Norm on the SRE18 dev set and SRE19 eval set. 
# The data directories are assumed to be 
#   $data/sre18_dev_enroll
#   $data/sre18_dev_test
#   $data/sre19_eval_enroll
#   $data/sre19_eval_test
# 
# $data should be set using `export data=...`
set -o pipefail
trap "" PIPE

cmd=run.pl
start_num=0
end_num=800

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 12 ]; then
  echo "Usage: $0 <mean> <lda> <whiten> <plda> <xvectors-cohort> <data-sre18-dev-enroll> <xvectors-sre18-dev-enroll> <xvectors-sre18-dev-test> <data-sre19-enroll> <xvectors-sre19-enroll> <xvectors-sre19-test> <score-dir>"
  echo ""

  exit 1;
fi

mean=$1
lda=$2
whiten=$3
plda=$4
xvectors_cohort=$5
data_dev_enroll=$6
xvectors_dev_enroll=$7
xvectors_dev_test=$8
data_eval_enroll=$9
xvectors_eval_enroll=${10}
xvectors_eval_test=${11}
score_dir=${12}

for file in $data_dev_enroll/spk2utt $data/sre18_dev_test/trials $data/sre18_dev_test/utt2spk $data_eval_enroll/spk2utt $data/sre19_eval_test/trials $data/sre19_eval_test/utt2spk ; do
  [ ! -f $file ] && echo "Expect file $file to exist!" && exit 1
done

mkdir -p $score_dir

# Scoring as usual
(
$cmd $score_dir/log/asnorm_score_sre18.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:$xvectors_dev_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 $plda - |" \
  "ark:ivector-mean ark:$data_dev_enroll/spk2utt scp:$xvectors_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $mean ark:- ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_dev_test/xvector.scp ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$data/sre18_dev_test/trials' | cut -d\  --fields=1,2 |" $score_dir/asnorm_score_sre18 || exit 1;

$cmd $score_dir/log/asnorm_score_sre19.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:$xvectors_eval_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 $plda - |" \
  "ark:ivector-mean ark:$data_eval_enroll/spk2utt scp:$xvectors_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $mean ark:- ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_eval_test/xvector.scp ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$data/sre19_eval_test/trials' | cut -d\  --fields=1,2 |" $score_dir/asnorm_score_sre19 || exit 1;
) &

# Scoring for the models
(
python backend_utils/generate_asnorm_list.py $data_dev_enroll/spk2utt $xvectors_cohort/xvector.scp $score_dir/asnorm_trials_sre18_enroll
$cmd $score_dir/log/asnorm_score_sre18_enroll.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:$xvectors_dev_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 $plda - |" \
  "ark:ivector-mean ark:$data_dev_enroll/spk2utt scp:$xvectors_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $mean ark:- ark:- | transform-vec $lda ark:- ark:- |  transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_cohort/xvector.scp ark:- | transform-vec $lda ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  $score_dir/asnorm_trials_sre18_enroll $score_dir/asnorm_score_sre18_enroll || exit 1;

python backend_utils/generate_asnorm_list.py $data_eval_enroll/spk2utt $xvectors_cohort/xvector.scp $score_dir/asnorm_trials_sre19_enroll
$cmd $score_dir/log/asnorm_score_sre19_enroll.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:$xvectors_eval_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 $plda - |" \
  "ark:ivector-mean ark:$data_eval_enroll/spk2utt scp:$xvectors_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $mean ark:- ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_cohort/xvector.scp ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  $score_dir/asnorm_trials_sre19_enroll $score_dir/asnorm_score_sre19_enroll || exit 1;
) &

# Scoring for the test
(
python backend_utils/generate_asnorm_list.py $data/sre18_dev_test/utt2spk $xvectors_cohort/xvector.scp $score_dir/asnorm_trials_sre18_test
$cmd $score_dir/log/asnorm_score_sre18_test.log \
  ivector-plda-scoring --normalize-length=true \
  "ivector-copy-plda --smoothing=0.0 $plda - |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_dev_test/xvector.scp ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_cohort/xvector.scp ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  $score_dir/asnorm_trials_sre18_test $score_dir/asnorm_score_sre18_test || exit 1;

python backend_utils/generate_asnorm_list.py $data/sre19_eval_test/utt2spk $xvectors_cohort/xvector.scp $score_dir/asnorm_trials_sre19_test
$cmd $score_dir/log/asnorm_score_sre19_test.log \
  ivector-plda-scoring --normalize-length=true \
  "ivector-copy-plda --smoothing=0.0 $plda - |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_eval_test/xvector.scp ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $mean scp:$xvectors_cohort/xvector.scp ark:- | transform-vec $lda ark:- ark:- | transform-vec $whiten ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  $score_dir/asnorm_trials_sre19_test $score_dir/asnorm_score_sre19_test || exit 1;
) &

wait 

python backend_utils/asnorm.py 1 $start_num $end_num $score_dir/asnorm_score_sre18_enroll $score_dir/asnorm_score_sre18_test $score_dir/asnorm_score_sre18 $score_dir/asnorm_score_sre18_norm
python backend_utils/asnorm.py 1 $start_num $end_num $score_dir/asnorm_score_sre19_enroll $score_dir/asnorm_score_sre19_test $score_dir/asnorm_score_sre19 $score_dir/asnorm_score_sre19_norm

