export LD_LIBRARY_PATH="/home/fq/anaconda3/envs/isaac/lib:$LD_LIBRARY_PATH"
python phc/run_hydra.py learning=im_big exp_name=debug_train_fix_disc_no_amp \
    env=env_im robot=smpl_humanoid \
    env.motion_file=/home/fq/repo/PHC/data/behave/Date01_Sub01_boxsmall_hand_upright.pkl \
    epoch=8000 test=True env.num_envs=1 headless=False no_virtual_display=True


#im_eval=True 