export LD_LIBRARY_PATH="/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH"
python phc/run_hydra.py learning=im_big exp_name=finetune_2D \
    env=env_im robot=smpl_humanoid \
    env.motion_file=/home/fengqiao/home/repo/PHC/data/behave/Date01_Sub01_boxsmall_hand_upright.pkl \
    epoch=8000 test=True env.num_envs=1 im_eval=True

# headless=False no_virtual_display=True