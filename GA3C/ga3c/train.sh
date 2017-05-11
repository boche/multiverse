### for CMU rocks cluster ###
#PBS -j oe
#PBS -k oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00
#PBS -N GA3C
#PBS -q gpu

module load gcc-4.9.2
module load cuda-8.0
export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`
source activate p_3

echo $HOSTNAME
echo "Running on gpu"
echo "Device = $CUDA_VISIBLE_DEVICES"

cd /home/bchen2/multiverse/GA3C/ga3c

#mkdir checkpoints > /dev/null 2>&1
#mkdir logs > /dev/null 2>&1
#python GA3C.py "$@"
#python GA3C.py DYNAMIC_SETTINGS=False CONCURRENT_EPISODES=1

#python GA3C.py CONCURRENT_EPISODES=4

