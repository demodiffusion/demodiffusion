import os
import numpy as np

import faulthandler
import time 
import hydra
from manimo.environments.single_arm_env import SingleArmEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from manimo.utils.transformations import *

import cv2

from PIL import Image


BASE_PATH = '../human_data'


DROID_CONTROL_FREQUENCY = 15
RETARGET_MODE = "cartesian_position"

# We follow configuration from Pi-0. Only difference is that we increase denoising step as 20 for both Pi-0 and DemoDiffusion.
open_loop_horizon = 8
predict_action_horizon = 10    
full_action_dim = 32
droid_action_dim = 8


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    # add task name 
    parser.add_argument("--task", type=str, default="closelaptop", help="task name")
    
    ## add traj num
    parser.add_argument("--traj", type=int, default=0, help="trajectory number")
        
    ## time denoise, how much noise to add from retargeted human demo.
    parser.add_argument("--time_denoise", type=float, default=1.0, help="time denoise")

    ## record 
    parser.add_argument("--record", action="store_true", help="if true, save videos")
        
    ## gripper threshold from human demo
    parser.add_argument("--gripper_threshold", type=float, default=0.2, help="gripper threshold")


    return parser.parse_args()

def main():
    args = parse_args()
    time_denoise = args.time_denoise


    hydra.initialize(config_path="manimo/manimo/conf", job_name="collect_demos_test")
    env_cfg = hydra.compose(config_name="env")
    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")


    # create a single arm environment
    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)

    env.reset()

    obs = env.get_obs()
    for key in obs:
        print(f"obs key: {key}, values: {obs[key]}")


    client = websocket_client_policy.WebsocketClientPolicy(host="128.2.194.118", port=8000)
    
    data_path = os.path.join(BASE_PATH, args.task, f"traj_{args.traj}")

    eef_pose_list = np.load(os.path.join(data_path, "processed_3d", "eef_pose.npy"))
    retarget_gripper_action_list = np.load(os.path.join(data_path, "processed_3d", "retarget_gripper_action.npy"))
    retarget_gripper_action_list =  np.where(retarget_gripper_action_list>args.gripper_threshold, 1, 0).astype(float)

    
    # initialize the robot
    start_time = time.time()
    init_eef_pose = np.zeros(7)    
    init_eef_pose[:3] = eef_pose_list[0][:3]
    init_eef_pose[3:6] = quat_to_euler(eef_pose_list[0][3:])
    robot_pose = np.array(env.actuators[0].get_robot_state()[0]['cartesian_position'])
    while not np.allclose(robot_pose, init_eef_pose[:6], atol=0.1) and time.time() - start_time < 20:        
        action_dict = env.actuators[0].create_action_dict(action=init_eef_pose, action_space="cartesian_position", gripper_action_space=None, robot_state=None)
        arm_action = np.array(action_dict["joint_velocity"]) * 0.5
        env.actuators[0].step(arm_action)
        robot_pose = np.array(env.actuators[0].get_robot_state()[0]['cartesian_position'])    
    env.actuators[1].step([retarget_gripper_action_list[0]])




    if args.record:
        from datetime import datetime
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        ori_str = "pose"

        SAVE_PATH = os.path.join(data_path, "paper")
        SAVE_PATH = os.path.join(SAVE_PATH, f"openpi_rollout/{time_denoise}/{RETARGET_MODE}/{ori_str}/{date_time_str}")
    

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)
            os.makedirs(SAVE_PATH+'/raw', exist_ok=True)
        


    exterior_images = []
    wrist_images = []
    exterior_images_raw = []
    wrist_images_raw = []
    
    print("Press c+Enter to start inference!!!!")
    import pdb; pdb.set_trace()
    timestep = 0
    while timestep < len(eef_pose_list) - predict_action_horizon:
        print("step")
        actions_demo = np.zeros((predict_action_horizon, full_action_dim))
        
        if timestep + predict_action_horizon < len(eef_pose_list):
            actions_demo[:,7] = retarget_gripper_action_list[timestep : timestep + predict_action_horizon]
 
        else:
            actions_demo[:,7] = retarget_gripper_action_list[-predict_action_horizon:]

        
        retarget_eef_pose_abs_next = np.zeros((predict_action_horizon, 7))
        retarget_eef_pose_abs_next[:,:3] = eef_pose_list[timestep+1: timestep + 1 + predict_action_horizon, :3]
        retarget_eef_pose_abs_next[:,3:6] = quat_to_euler(eef_pose_list[timestep+1: timestep + 1 + predict_action_horizon, 3:])
        
        
        for i in range(predict_action_horizon):            
            feed = retarget_eef_pose_abs_next[i] 
            
            if i == 0:
                action_dict = env.actuators[0].create_action_dict(action=feed, action_space=RETARGET_MODE, gripper_action_space=None, robot_state=None)     
                
            else:
                action_dict = env.actuators[0].create_action_dict(action=feed, action_space=RETARGET_MODE, gripper_action_space=None, robot_state=robot_state)     

            robot_state = action_dict.pop("robot_state")
            for key in action_dict:
                robot_state[key] = action_dict[key]

            actions_demo[i,:7] = action_dict["joint_velocity"] 
                    
        
        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        start_time = time.time()
        if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= open_loop_horizon:
            actions_from_chunk_completed = 0
            

            obs = env.get_obs()
            
            observation = {
                "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs['cam0'][0], 224, 224)
                ),
                "observation/wrist_image_left": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs['cam_wrist_left'][0], 224, 224)
                ),
                "observation/joint_position": obs['q_pos'],
                "observation/gripper_position": np.array([obs['eef_gripper_width']]),

                # Text Prompt examples.
                # "prompt": "wipe the table",                
                # "prompt": "drag the basket to right",
                # "prompt": "pick and place the banana on the plate",
                "prompt": "pick and place the bowl on the stove",
                # "prompt": "close the microwave",                
                # "prompt": "move the ironing machine to the right",
                # "prompt": "pick up the bear doll",
                # "prompt": "close the laptop",
                
                "retargeted_actions": actions_demo,
                "time": time_denoise,
                "action_dim_used" : droid_action_dim  # deprecated: hardcoded to 8 in pi0.py due to JAX JIT issues
            }

            pred_action_chunk = client.infer(observation)["actions"][:, :droid_action_dim]
            
            
            if args.record:
                exterior_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs['cam0'][0], 224, 224)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs['cam_wrist_left'][0], 224, 224)
                )

                exterior_images.append(exterior_img)
                wrist_images.append(wrist_img)
                exterior_images_raw.append(obs['cam0'][0])
                wrist_images_raw.append(obs['cam_wrist_left'][0])


            
        action = pred_action_chunk[actions_from_chunk_completed]
        actions_from_chunk_completed += 1
        

        
        # Binarize gripper action
        if action[-1].item() > 0.5:
            # action[-1] = 1.0
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            # action[-1] = 0.0
            action = np.concatenate([action[:-1], np.zeros((1,))])


        
        
        arm_action = np.array(action[:7]) 
        gripper_action = np.array([action[-1]])
            

        env.step([arm_action, gripper_action])
        timestep += 1
        
        elapsed_time = time.time() - start_time
        print(1/elapsed_time)
        if elapsed_time < 1/ DROID_CONTROL_FREQUENCY:
            time.sleep(1/ DROID_CONTROL_FREQUENCY - elapsed_time)

        
    if args.record:
        for idx, (exterior_img, wrist_img) in enumerate(zip(exterior_images, wrist_images)):
            Image.fromarray(exterior_img).save(os.path.join(SAVE_PATH, f"exterior_{idx:04d}.png"))
            Image.fromarray(wrist_img).save(os.path.join(SAVE_PATH, f"wrist_{idx:04d}.png"))

        for idx, (exterior_img_raw, wrist_img_raw) in enumerate(zip(exterior_images_raw, wrist_images_raw)):
            Image.fromarray(exterior_img_raw).save(os.path.join(SAVE_PATH +'/raw', f"exterior_{idx:04d}.png"))
            Image.fromarray(wrist_img_raw).save(os.path.join(SAVE_PATH +'/raw', f"wrist_{idx:04d}.png"))        
    

    print("Inference Done!!! Press c+enter to reset the robot.")
    import pdb; pdb.set_trace()
    
    env.actuators[1].step([0]) # open the gripper
    env.reset()
    
    


if __name__ == "__main__":
    main()