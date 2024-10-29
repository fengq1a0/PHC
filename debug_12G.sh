export LD_LIBRARY_PATH="/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH"
python phc/run_hydra.py \
 learning=debug \
 exp_name=debug \
 env=env_im robot=smpl_humanoid \
 env.motion_file=/home/fengqiao/home/repo/PHC/data/behave/Date01_Sub01_boxsmall_hand_upright.pkl \
 epoch=7500 \
 env.num_envs=16 \