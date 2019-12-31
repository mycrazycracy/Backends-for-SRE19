if [ `hostname` == 'a12' ]; then
    export KALDI_ROOT=/mnt/a12/liuyi/software/kaldi_gpu
elif [ `hostname` == 'a11' ]; then
    export KALDI_ROOT=/mnt/a12/liuyi/software/kaldi_gpu_cuda9
else
    export KALDI_ROOT=/mnt/a12/liuyi/software/kaldi_cpu
fi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
export LD_LIBRARY_PATH=/home/liuyi/software/anaconda2/lib:/home/liuyi/cudnn7/lib64:$LD_LIBRARY_PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
