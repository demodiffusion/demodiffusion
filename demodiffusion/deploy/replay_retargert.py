import os 
import numpy as np

import faulthandler
import time 
import hydra
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.utils.transformations import *

from openpi_client import image_tools

import cv2
from PIL import Image


BASE_PATH = '../human_data'


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    # add task name 
    parser.add_argument("--task", type=str, default="closelaptop", help="task name")
    ## add traj num
    parser.add_argument("--traj", type=int, default=0, help="trajectory number")
   
    ## record 
    parser.add_argument("--record", action="store_true", help="if true, save videos")

    ## gripper threshold from human demo
    parser.add_argument("--gripper_threshold", type=float, default=0.2, help="gripper threshold")
    

    return parser.parse_args()


def main():
    args = parse_args()

    # create a single arm environment
    DROID_CONTROL_FREQUENCY = 15

    hydra.initialize(config_path="manimo/manimo/conf", job_name="collect_demos_test")
    env_cfg = hydra.compose(config_name="env")
    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")

    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
    env.reset()

    data_path = os.path.join(BASE_PATH, args.task, f"traj_{args.traj}")



    if args.record:
        from datetime import datetime
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        ori_str = "pose"

        # for recording
        SAVE_PATH = os.path.join(data_path, "paper")
        SAVE_PATH = os.path.join(SAVE_PATH, f"kinematic_rollout/{ori_str}/{date_time_str}")

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)    
            os.makedirs(SAVE_PATH + "/raw", exist_ok=True)    


    eef_pose_list = np.load(os.path.join(data_path, "processed_3d", "eef_pose.npy"))
    retarget_gripper_action_list = np.load(os.path.join(data_path, "processed_3d", "retarget_gripper_action.npy"))


    init_joint_pos = []
    retarget_arm_action_list = []
    retarget_gripper_action_list_final = []
    retarget_eef_pose_list = []
    exterior_images = []
    wrist_images = []
    exterior_images_raw = []
    wrist_images_raw = []
    episode_length = len(eef_pose_list) 

    for i in range(episode_length):

        print("step")
        start_time = time.time()

        
        eef_pose = np.zeros(7)
        eef_pose[:3] = eef_pose_list[i][:3]
        eef_pose[3:6] = quat_to_euler(eef_pose_list[i][3:])                
        retarget_gripper_action = retarget_gripper_action_list[i:i+1]
        if retarget_gripper_action[0] > args.gripper_threshold: # This is the threshold for gripper action. Tune this if gripper is not activated well.
            retarget_gripper_action_execute = [1]
        else:
            retarget_gripper_action_execute = [0]

        
        if i == 0 :
            # before saving inital joint position, we need to initialize the robot
            robot_pose = np.array(env.actuators[0].get_robot_state()[0]['cartesian_position'])
            while not np.allclose(robot_pose, eef_pose[:6], atol=0.1) and time.time() - start_time < 20:        
                action_dict = env.actuators[0].create_action_dict(action=eef_pose, action_space="cartesian_position", gripper_action_space=None, robot_state=None)
                arm_action = np.array(action_dict["joint_velocity"]) * 0.5
                env.actuators[0].step(arm_action)
                robot_pose = np.array(env.actuators[0].get_robot_state()[0]['cartesian_position'])
            
                env.actuators[1].step(retarget_gripper_action_execute)

            init_joint_pos = env.get_obs()['q_pos'].copy()


                        
            print("Initializing done, press c+enter to proceed.")
            timestep = 0
            import pdb; pdb.set_trace()            

        
        
        else: 
            timestep += 1
            if args.record:

                obs = env.get_obs()

                exterior_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs['cam0'][0], 224, 224)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs['cam_wrist_left'][0], 224, 224)
                )
                
                exterior_images.append(exterior_img)
                exterior_images_raw.append(obs['cam0'][0])
                wrist_images.append(wrist_img)
                wrist_images_raw.append(obs['cam_wrist_left'][0])
                


                
            
            robot_pose = np.array(env.actuators[0].get_robot_state()[0]['cartesian_position'])
            action_dict = env.actuators[0].create_action_dict(action=eef_pose, action_space="cartesian_position", gripper_action_space=None, robot_state=None)
            arm_action = np.array(action_dict["joint_velocity"]) * 1.0
            env.step([arm_action, retarget_gripper_action_execute])
            
        
            retarget_arm_action_list.append(arm_action)
            retarget_gripper_action_list_final.append(retarget_gripper_action_execute)
            retarget_eef_pose_list.append(action_dict["cartesian_position"]) 
            


        elapsed_time = time.time() - start_time
        print(1/elapsed_time)
        if elapsed_time < 1/ DROID_CONTROL_FREQUENCY:
            time.sleep(1/ DROID_CONTROL_FREQUENCY - elapsed_time)



    # After the loop
    if args.record:
        for idx, (exterior_img, wrist_img) in enumerate(zip(exterior_images, wrist_images)):
            Image.fromarray(exterior_img).save(os.path.join(SAVE_PATH, f"exterior_{idx:04d}.png"))
            Image.fromarray(wrist_img).save(os.path.join(SAVE_PATH, f"wrist_{idx:04d}.png"))

        for idx, (exterior_img_raw, wrist_img_raw) in enumerate(zip(exterior_images_raw, wrist_images_raw)):
            Image.fromarray(exterior_img_raw).save(os.path.join(SAVE_PATH +'/raw', f"exterior_{idx:04d}.png"))
            Image.fromarray(wrist_img_raw).save(os.path.join(SAVE_PATH +'/raw', f"wrist_{idx:04d}.png"))

    print("Eval Done! Press c+enter to reset.")
    import pdb; pdb.set_trace()
    env.actuators[1].step([0]) # open the gripper
    env.reset()

    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()