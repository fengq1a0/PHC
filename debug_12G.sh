export LD_LIBRARY_PATH="/home/fq/anaconda3/envs/isaac/lib:$LD_LIBRARY_PATH"

tmp="debug_rendering_reward"

mkdir output/HumanoidIm/$tmp
cp output/HumanoidIm/debug/Humanoid_00007500.pth output/HumanoidIm/$tmp/

python phc/run_hydra.py \
 learning=debug \
 exp_name=$tmp \
 env=env_im robot=smpl_humanoid \
 env.motion_file=/home/fq/repo/PHC/data/behave/Date01_Sub01_boxsmall_hand_upright.pkl \
 epoch=7500 \
 env.num_envs=256 \
 finetune_wRR=True \
# fix_disc=True \
# position_only=True \
# pos_2d_only=True \

# env.motion_file=/home/fq/repo/PHC/data/behave/Date01_Sub01_boxsmall_hand_upright.pkl \
# env.motion_file=/home/fq/repo/PHC/data/amass_smpl/debug.pkl \