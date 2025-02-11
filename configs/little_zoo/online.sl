#!/bin/bash
#SBATCH --job-name=online_alp   # job name
#SBATCH --time=40:00:00 # maximum execution time (HH:MM:SS)
#SBATCH --output=your_path    # output file name
#SBATCH --error=your_path%a.err      # err file name
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH -C h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --account=your_account

#SBATCH --array=0-7

module purge
module load arch/h100
module load python/3.11.5
conda activate glam

MASTER_PORT=$((18100+SLURM_ARRAY_TASK_ID))

srun python -m lamorel_launcher.launch --config-path $WORK/MAGELLAN-ICML/configs/little_zoo/ --config-name local_gpu_config_online rl_script_args.path=$WORK/MAGELLAN-ICML/magellan/main.py rl_script_args.output_dir=your_path rl_script_args.seed=${SLURM_ARRAY_TASK_ID} lamorel_args.accelerate_args.main_process_port=${MASTER_PORT} #rl_script_args.loading_path=your_path