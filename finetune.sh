export LD_LIBRARY_PATH="/home/fq/anaconda3/envs/isaac/lib:$LD_LIBRARY_PATH"
python phc/run_hydra.py learning=im_big exp_name=finetune_2D env=env_im robot=smpl_humanoid \
    env.motion_file=/home/fq/repo/PHC/data/behave/Date01_Sub01_boxsmall_hand_upright.pkl \
    epoch=7500 finetune_wRR=True position_only=True