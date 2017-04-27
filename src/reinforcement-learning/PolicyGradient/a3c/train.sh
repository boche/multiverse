### for CMU rocks cluster ###
#PBS -j oe
#PBS -k oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00
#PBS -N A3C
#PBS -q gpu 

module load gcc-4.9.2
module load cuda-8.0
export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`
source activate p_3

echo $HOSTNAME
echo "Running on gpu"
echo "Device = $CUDA_VISIBLE_DEVICES"

cd /home/bchen2/multiverse/reinforcement-learning/PolicyGradient/a3c
python train.py --model_dir /home/bchen2/tmp/a3c-test --env PongDeterministic-v0 --t_max 5 --eval_every 120 --parallelism 16 
