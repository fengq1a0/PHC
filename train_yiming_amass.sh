export LD_LIBRARY_PATH="/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH"
python phc/run_hydra.py learning=im_big exp_name=amass_prim env=env_im robot=smpl_humanoid \
 env.motion_file=data/amass_smpl/train.pkl \
 epoch=6000 resume_str=hmxodne1

# Used to pre-train
# change epoch to resume
# change resume_str to continue wandb Log
