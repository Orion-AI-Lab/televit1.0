#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

wandb_project="your-wandb-project"
experiment="unet_nots"
max_epochs=30
batch_size=128

for target_shift in 16 0 8 1 4 2
do
  echo "Experiment with target_shift=${target_shift}"
  python src/train.py target_shift=${target_shift} data.debug=${debug} trainer.max_epochs=${max_epochs} data.batch_size=${batch_size} logger=wandb logger.wandb.project=${wandb_project} model.loss=ce model.encoder="efficientnet-b1" experiment=${experiment}
done