#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

max_epochs=30
wandb_project="your-wandb-project"
batch_size=128
experiment="televit_nots"
heads=8
mlp_dim=1536
depth=8

for target_shift in 1 8 4 2 0 16
do
  tag="plain_vit"
  python src/train.py ++model.heads=${heads} ++model.depth=${depth} ++model.mlp_dim=${mlp_dim} ++model.input_global_shape="[]" ++model.patch_global_shape="[]" ++data.patch_global_shape="[]" ++model.input_oci_shape="[]" ++model.patch_oci_shape="[]" data.batch_size=${batch_size} target_shift=${target_shift} trainer.max_epochs=${max_epochs} logger=wandb experiment=${experiment} logger.wandb.project=${wandb_project} tags="[${experiment}, ${tag}]"

  tag="televit_oci"
  python src/train.py ++model.heads=${heads} ++model.depth=${depth} ++model.mlp_dim=${mlp_dim} ++model.input_global_shape="[]" ++model.patch_global_shape="[]" ++data.patch_global_shape="[]" data.batch_size=${batch_size} target_shift=${target_shift} trainer.max_epochs=${max_epochs} logger=wandb experiment=${experiment} logger.wandb.project=${wandb_project} tags="[${experiment}, ${tag}]"

  tag="televit_global"
  python src/train.py ++model.heads=${heads} ++model.depth=${depth} ++model.mlp_dim=${mlp_dim} ++model.input_oci_shape="[]" ++model.patch_oci_shape="[]"  data.batch_size=${batch_size} target_shift=${target_shift} trainer.max_epochs=${max_epochs} logger=wandb experiment=${experiment} logger.wandb.project=${wandb_project} tags="[${experiment}, ${tag}]"

  tag="televit_full"
  python src/train.py ++model.heads=${heads} ++model.depth=${depth} ++model.mlp_dim=${mlp_dim}  data.batch_size=${batch_size} target_shift=${target_shift} trainer.max_epochs=${max_epochs} logger=wandb experiment=${experiment} logger.wandb.project=${wandb_project} tags="[${experiment}, ${tag}]"
done


