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
    base_path = "/home/fq/Downloads/behave/processed/"
    name_list = [
        "Date01_Sub01_boxsmall_hand",
    ]
    R = np.array([[1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0], 
                  [0.0, -1.0, 0.0]])
    #------------------------------------------------------
    behave_full_motion_dict = {}


    for name in tqdm(name_list):
        data_path = os.path.join(base_path, name)
        smpl_motion = np.load(os.path.join(data_path, "smpl_fit_all.npz"))
        object_motion = np.load(os.path.join(data_path, "object_fit_all.npz"))

        bound = 300           # number of frames to use.
        framerate = 30        # framerate of the sequence.

        skip = int(framerate/30)
        root_trans = smpl_motion['trans'][::skip, :].astype(np.float64)
        pose_aa = np.concatenate([smpl_motion['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)#smpl_motion['poses'][::skip, :]#

        #-------------------camera to world-------------------------
        # camera coordinate to ...
        rot_mat = sRot.from_rotvec(pose_aa[:,0:3]).as_matrix()
        rot_mat = np.matmul(R, rot_mat)
        pose_aa[:, 0:3] =  sRot.from_matrix(rot_mat).as_rotvec()
        # also root trans!
        root_trans = root_trans @ R.T
        #-----------------------------------------------------------


        bound = min(bound, root_trans.shape[0])
        N = bound
        root_trans = root_trans[:bound]
        pose_aa = pose_aa[:bound]
    
        smplh_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

        # always use zero shape and neutral model here
        beta = np.zeros((16))
        gender_number, beta[:], gender = [0], 0, "neutral"
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

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
        fps = 30

        new_motion_out = {}
        new_motion_out['pose_quat_global'] = pose_quat_global
        new_motion_out['pose_quat'] = pose_quat
        new_motion_out['trans_orig'] = root_trans
        new_motion_out['root_trans_offset'] = root_trans_offset
        new_motion_out['beta'] = beta
        new_motion_out['gender'] = gender
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['fps'] = fps
        behave_full_motion_dict[name] = new_motion_out

os.makedirs("data/behave", exist_ok=True)
if upright_start:
    joblib.dump(behave_full_motion_dict, "data/behave/Date01_Sub01_boxsmall_hand_upright.pkl", compress=True)
else:
    joblib.dump(behave_full_motion_dict, "data/behave/behave_take0_take6.pkl", compress=True)