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
    model_head = torch.jit.load("/home/fq/repo/4D-Humans/hmr2_head.pt")
    debug_data = True
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
    base_path = "/home/fq/data/behave/processed/"
    if process_split == "train":
        with open("/home/fq/data/behave/scripts/data_list_train.txt", "r") as fi:
            name_list = fi.read().split()
    else:
        with open("/home/fq/data/behave/scripts/data_list_test.txt", "r") as fi:
            name_list = fi.read().split()
    if debug_data:
        name_list = name_list[:16]




    ground_normals = np.load("/home/fq/data/behave/grounds/planes.npy")
    smpl = SMPL(model_path="/home/fq/repo/PHC/data/smpl/", gender="neutral")
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    #------------------------------------------------------
    behave_full_motion_dict = {}


    for name in tqdm(name_list):
        data_path = os.path.join(base_path, name)
        smpl_motion = np.load(os.path.join(data_path, "refit_smpl.npz"))

        trans = smpl_motion['trans'].astype(np.float64)
        poses = smpl_motion['poses'].astype(np.float64)
        betas = smpl_motion['betas'].astype(np.float64)
        if debug_data:
            trans = smpl_motion['trans'].astype(np.float64)#[:150]
            poses = smpl_motion['poses'].astype(np.float64)#[:150]
            betas = smpl_motion['betas'].astype(np.float64)#[:150]

        with torch.no_grad():
            j0 = smpl(
                betas=torch.from_numpy(betas).float(),
                global_orient=torch.zeros((betas.shape[0], 3)),
                body_pose=torch.zeros((betas.shape[0], 23*3))
            ).joints[:,0].numpy()

        g_m = ground_normals[int(name[5])]
        y_m = g_m[:3]
        p_m = np.array([0, -g_m[3] / g_m[1], 0, 1])

        Ti = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        RRRR = np.array([[ 1.0,  0.0,  0.0, 0],
                         [ 0.0,  0.0, -1.0, 0], 
                         [ 0.0,  1.0,  0.0, 0],
                         [   0,    0,    0, 1]])

        for i in [0, 1, 2, 3]:
            #########################################################################################################
            # For FQ features
            with open(os.path.join(data_path, "intri/%d/calibration.json" % i), "r") as tmp:
                data = json.load(tmp)
                cx =    np.array(data["color"]["cx"])
                cy =    np.array(data["color"]["cy"])
                fx =    np.array(data["color"]["fx"])
                fy =    np.array(data["color"]["fy"])
                width = np.array(data["color"]["width"])

            image_feature = np.load(os.path.join(data_path, "fq_info_%d.npz" % i))
            
            # Regulize kp2d and bbox
            bbox = image_feature["bbox"].copy()
            bbox[:,0] = (bbox[:,0]-cx) / fx *2.8
            bbox[:,1] = (bbox[:,1]-cy) / fy *2.8
            bbox[:,2] = (bbox[:,2] - 0.12 * (fx+fy)) / (0.3 * (fx + fy)) # 0.03
            kp2d = image_feature["kp2d"].copy()
            kp2d[:,:,0] = (kp2d[:,:,0]-cx) / fx *2.8
            kp2d[:,:,1] = (kp2d[:,:,1]-cy) / fy *2.8
            img_feat = image_feature["img_feat"]
            with torch.no_grad():
                TRAM_pose, TRAM_shape = model_head(torch.from_numpy(img_feat).to("cuda:0"))
                TRAM = torch.cat([TRAM_pose, TRAM_shape],dim=1).cpu().numpy()
            #########################################################################################################
            # For T_m2c, T_c2g
            with open(os.path.join(data_path, "extri/%d/config.json" % i), "r") as tmp:
                data = json.load(tmp)
                R = np.array(data["rotation"]).reshape(3,3)
                T = np.array(data["translation"])
                T_m2c = np.eye(4)
                T_m2c[:3,:3] = R
                T_m2c[:3,3] = T
                T_m2c = np.linalg.inv(T_m2c)
                T_m2c = Ti@T_m2c@Ti
            
            y_c = T_m2c[:3,:3] @ y_m
            y_c = y_c / np.linalg.norm(y_c)
            x_c = np.cross(y_c, np.array([0.0, 0.0, 1.0]))
            x_c = x_c / np.linalg.norm(x_c)
            z_c = np.cross(x_c, y_c)
            z_c = z_c / np.linalg.norm(z_c)
            R_c2g = np.array([x_c, y_c, z_c])

            p_g = R_c2g @ (T_m2c @ p_m)[:3]
            T_c2g = np.eye(4)
            T_c2g[:3,:3] = R_c2g
            T_c2g[1,3] = - p_g[1]

            T_c2g_feature = np.zeros(10,dtype=np.float32)
            T_c2g_feature[:9] = R_c2g.reshape(9)
            T_c2g_feature[9] = - p_g[1]
            #########################################################################################################
            # For isaac gym:
            # poses_isaac, trans_isaac, betas
            T_m2g = RRRR @ T_c2g @ T_m2c

            poses_isaac = poses.copy()
            root_rot = sRot.from_rotvec(poses_isaac[:,:3]).as_matrix()
            poses_isaac[:,:3] = sRot.from_matrix(T_m2g[:3,:3] @ root_rot).as_rotvec()

            trans_isaac = (trans + j0) @ T_m2g[:3,:3].T + T_m2g[:3,3] - j0


            # For original smpl:
            # poses_g, trans_g, betas
            T_m2g = T_c2g @ T_m2c

            poses_g = poses.copy()
            root_rot = sRot.from_rotvec(poses_g[:,:3]).as_matrix()
            poses_g[:,:3] = sRot.from_matrix(T_m2g[:3,:3] @ root_rot).as_rotvec()

            trans_g = (trans + j0) @ T_m2g[:3,:3].T + T_m2g[:3,3] - j0
            #########################################################################################################
            N = min(poses_isaac.shape[0], kp2d.shape[0])
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
            new_motion_out['T_c2g'] = np.tile(T_c2g_feature[None], (N,1))   # R + height
            new_motion_out["kp2d"] = kp2d[:N].reshape(N,23*3)               # 17 + 6 (two feet)     # Need to be regulized
            new_motion_out["bbox"] = bbox[:N]                                                       # Need to be regulized
            new_motion_out["img_feat"] = img_feat[:N]
            # additional
            new_motion_out['fps']     = 30
            new_motion_out['gender']  = "neutral"
            new_motion_out['betas']   = betas
            new_motion_out['poses_g'] = poses_g
            new_motion_out['trans_g'] = trans_g

            new_motion_out["TRAM"]    = TRAM[:N]
            

            behave_full_motion_dict[name+"_%d" % i] = new_motion_out

os.makedirs("data/behave_ground", exist_ok=True)
if upright_start:
    if debug_data:
        joblib.dump(behave_full_motion_dict, "data/behave_ground/behave_debug.pkl", compress=True)
    elif process_split == "train":
        joblib.dump(behave_full_motion_dict, "data/behave_ground/behave_train.pkl", compress=True)
    else:
        joblib.dump(behave_full_motion_dict, "data/behave_ground/behave_test.pkl", compress=True)