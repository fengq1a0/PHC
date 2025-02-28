import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
import cv2
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

import json
from smplx import SMPL


if __name__ == "__main__":
    #----------------------------------------------------
    process_split = "train"
    upright_start = True
    robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }
    smpl_local_robot = LocalRobot(robot_cfg,)
    #------------------------------------------------------
    base_path = "/home/fq/repo/GVHMR/"
    name_list = sorted(os.listdir(os.path.join(base_path, "EMDB2")))

    smpl = SMPL(model_path="/home/fq/repo/PHC/data/smpl/", gender="neutral")
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    #------------------------------------------------------
    behave_full_motion_dict = {}


    for name in tqdm(name_list):
        #if "P2_19_indoor_walk_off_mvs" in name:
        #    continue
        data_path = os.path.join(base_path, "EMDB2", name)
        smpl_motion = joblib.load(data_path)
        transl        = smpl_motion['transl'].astype(np.float64)[60:]
        body_pose     = smpl_motion['body_pose'].astype(np.float64)[60:]
        global_orient = smpl_motion['global_orient'].astype(np.float64)[60:]
        betas         = smpl_motion['betas'].astype(np.float64)[60:]

        betas = betas.mean(axis = 0)[None]

        tmp = transl.max(axis = 0) - transl.min(axis = 0)
        if tmp[1] < 0.3 and tmp[1] > 0.7:
            print(name)
            continue


        with torch.no_grad():
            j0 = smpl(
                betas=torch.from_numpy(betas).float(),
                global_orient=torch.zeros((betas.shape[0], 3)),
                body_pose=torch.zeros((betas.shape[0], 23*3))
            ).joints[:,0].numpy()



        image_feature = smpl_motion["feature"][0][60:]

        RRRR = np.array([[ 1.0,  0.0,  0.0, 0],
                         [ 0.0,  0.0, -1.0, 0], 
                         [ 0.0,  1.0,  0.0, 0],
                         [   0,    0,    0, 1]])


        #########################################################################################################
        # For isaac gym:
        # poses_isaac, trans_isaac, betas
        poses_isaac = np.zeros((body_pose.shape[0], 24*3), dtype=np.float64)
        poses_isaac[:,3:] = body_pose
        root_rot = sRot.from_rotvec(global_orient).as_matrix()
        poses_isaac[:,:3] = sRot.from_matrix(RRRR[:3,:3] @ root_rot).as_rotvec()
        trans_isaac = (transl + j0) @ RRRR[:3,:3].T + RRRR[:3,3] - j0


        # For original smpl:
        # poses_g, trans_g, betas
        poses_g = np.zeros((body_pose.shape[0], 24*3), dtype=np.float64)
        poses_g[:,3:] = body_pose
        poses_g[:,:3] = global_orient
        trans_g = transl
        #########################################################################################################
        N = min(poses_isaac.shape[0], trans_g.shape[0])
        poses_isaac = poses_isaac[:N]
        trans_isaac = trans_isaac[:N]

        # to mujoco
        pose_aa_mj = poses_isaac.reshape(N, 24, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)
        # smpl_local_robot?
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(betas), gender=[0], objs_info=None)
        smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        root_trans_offset = torch.from_numpy(trans_isaac) + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True)

        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()
        pose_quat_global = new_sk_state.global_rotation.numpy()
        pose_quat = new_sk_state.local_rotation.numpy()
        #########################################################################################################

        new_motion_out = {}
        new_motion_out['pose_quat_global'] = pose_quat_global
        new_motion_out['pose_quat'] = pose_quat
        new_motion_out['root_trans_offset'] = root_trans_offset
        new_motion_out['pose_aa'] = poses_isaac
        # FQ features
        new_motion_out["img_feat"] = image_feature[:N]
        # additional
        new_motion_out['fps']     = 30
        new_motion_out['gender']  = "neutral"
        new_motion_out['betas']   = betas
        new_motion_out['poses_g'] = poses_g
        new_motion_out['trans_g'] = trans_g

        behave_full_motion_dict[name.split()[0]] = new_motion_out

os.makedirs("data/behave_ground", exist_ok=True)
if upright_start:
    joblib.dump(behave_full_motion_dict, "data/EMDB2/EMDB2_GT.pkl", compress=True)