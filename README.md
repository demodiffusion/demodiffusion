# [DemoDiffusion](https://demodiffusion.github.io/)

<!-- Teaser Video --> 
<p align="left">
  <img src="teaser.gif" alt="Teaser Animation" width="480"/>
</p>


<a href="https://demodiffusion.github.io/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2506.20668"><strong>arXiv</strong></a>
  
  <!-- <a href="https://demodiffusion.github.io/"><strong>Data</strong></a> -->
  

  <a href="https://rureadyo.github.io/">Sungjae Park</a>, 
  <a href="https://homangab.github.io/">Homanga Bharadhwaj</a>, 
  <a href="https://shubhtuls.github.io/">Shubham Tulsiani</a>

Authors' implementation of DemoDiffusion.

This repository consists of two parts.

- deploy: Given a single human demonstration, perform inference of DemoDiffusion
- collect: Here we briefly explain how we collected the human demonstration, but deployment is separate as long as you satisfy requirements below.

<br>

# 💻 Installation
## Hardware Requirements 
We follow [DROID](https://droid-dataset.github.io/) setup with minimal change. Other than below, please follow instructions(hardware setup, and configuring the Franka robot part of software setup) in [DROID](https://droid-dataset.github.io/droid/). 


- Instead of Zed 2 for the rear view camera, we use Realsense D455 camera. 
- As the laptop used for DROID does not have powerful GPUs, we instead use a separate server/workstation for remote inferece, as sugggested in [Pi-0](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md).
    - Hence, we have 1. NUC for real time robot control, 2. DROID laptop for sending the robot control command to NUC, and 3. separate workstation/server for DemoDiffusion inference. These are referred as NUC, laptop, and workstation in the following.


## Environment Installation
1. First, clone the repository and its submodules. NUC requires manimo, DROID laptop requires manimo and part of openpi(for remote inference), and workstation requires openpi(optionally hamer for data collection).

    ```
    git clone --recursive https://github.com/demodiffusion/demodiffusion.git
    ```

2. To support more modular sensor inputs(i.e. switch Zed 2 to Realsense), we use [ManiMo](https://github.com/RUreadyo/manimo) instead of DROID for controlling the robot. Follow the instructions in [ManiMo](https://github.com/AGI-Labs/manimo) for installing the robot setup (NUC, DROID laptop). Make sure to use the cloned submodule, as we have some changes for DemoDiffusion.

3. After, on DROID laptop, run below.

    ```
    cd $PATH_TO_DEMODIFFUSION/demodiffusion/deploy/openpi/packages/openpi-client

    conda activate manimo-latest

    pip install -e .
    ```



4. Follow instructions in [Pi-0](https://github.com/RUreadyo/openpi) to install Pi-0 on the workstation. Again, make sure to use the cloned submodule, as we have some changes for DemoDiffusion.

## Human Demonstration Requirements 
We assume the human demonstration consists of 3D positions of hand keypoints. with the dimension of T (Length of Episode) * 21 (number of keypoints) * 3 (positions of each keypoint, represented in robot frame). We provide example data here.


### Folder Structure
Once you have collected your own dataset for the target task (for example, "close the laptop"), please organize your data using the following folder structure and put it in DROID laptop:

``` 
human_data
    └── closelaptop/
        └── traj_0/processed_3d/righthand_3d_keypoints.npy
        └── traj_1/processed_3d/righthand_3d_keypoints.npy
        └── traj_2/processed_3d/righthand_3d_keypoints.npy
```

- **human_data**: Root directory for all human demonstration data.
- **closelaptop**: Subdirectory for the "close the laptop" task.
- **traj_0, traj_1, traj_2, ...**: Each folder contains processed 3D keypoint data from a single trajectory or demonstration.

- **righthand_3d_keypoints.npy**: Processed 3D keypoint data from a single human trajectory or demonstration.

Add additional `traj_*` folders as needed for more demonstrations.

### Kinematic Retargeting
Once you have human demonstration, save kinematically retargeted robot end effector poses. In DROID laptop, run
 

    cd $PATH_TO_DEMODIFFUSION/demodiffusion/deploy
    
    conda activate manimo-latest
    
    python preprocess.py --task_name $TASK_NAME --traj_num $TRAJ_NUM 


<br>

# 🤖 Deploy DemoDiffusion
0. In DROID laptop, update config files (manimo/manimo/conf/arm/franka_arm_pizero.yaml, manimo/manimo/conf/camera/multi_real_sense_pizero.yaml) with your ip and camera id.

1. Turn on Franka Panda, NUC, and DROID laptop. 

    In NUC, run
    
    ```
    cd $PATH_TO_DEMODIFFUSION/demodiffusion/deploy/manimo/monometis/launcher/

    conda activate manimo-latest
    
    sudo pkill -9 run_server # to kill any existing servers
    
    ./launch_robot.sh
    ```

    Again in NUC, run 
    
    ```
    cd $PATH_TO_DEMODIFFUSION/demodiffusion/deploy/manimo/monometis/launcher/

    conda activate manimo-latest

    ./launch_gripper.sh
    ```



2. Turn on the workstation and enable remote inference of Pi-0.

    In workstation, run
    ```
    cd $PATH_TO_DEMODIFFUSION/demodiffusion/deploy/openpi
    
    python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid  --policy.dir=s3://openpi-assets/checkpoints/pi0_droid
    ```


3. Run DemoDiffusion Inference. During inference, at each timestep, we index kinematically retargeted end effector poses with same timestep. Then, it gets converted into joint velocity (action space of Pi-0 DROID) using inverse-kinematics. 

    In DROID laptop, run
    ```   
    cd PATH_TO_DEMODIFFUSION/demodiffusion/deploy
    conda activate manimo-latest
    
    python demodiffusion.py --task $TASK_NAME --traj $TRAJ_NUM --time_denoise NOISE_LEVEL 
    ```

    - We use noise level 0.2 (for tasks where Pi-0 completely fails) and 0.4 (for tasks where Pi-0 show nonzero success rate) in the paper.

    - To record the rollout, add --record.

    - To adjust threshold for kinematically retargeted gripper actions, set --gripper_threshold. By default, we use 0.2.

4. (Optional) To check the quality of kinematic retargeting only, run below in DROID laptop.
    ```   
    cd PATH_TO_DEMODIFFUSION/demodiffusion/deploy
    conda activate manimo-latest
    
    python replay_retarget.py --task $TASK_NAME --traj $TRAJ_NUM  
    ```

<br>

# 📊 Collect Your Own Human Demonstration
As long as the human demonstration consists of 3D hand keypoints as aforementioned, you can use it for deployment. We provide how we collected the human demonstration as one guideline [here](collect.md).


<br>

# 🙏 Acknowledgements
Our code is based on: [Pi-0](https://github.com/Physical-Intelligence/openpi), [Hamer](https://github.com/geopavlakos/hamer), and [Manimo](https://github.com/AGI-Labs/manimo). We thank all these authors for their open-sourced code.

<br>

# 📝 Citation

If you find our work useful, please consider citing:
```
@misc{park2025demodiffusiononeshothumanimitation,
      title={DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy}, 
      author={Sungjae Park and Homanga Bharadhwaj and Shubham Tulsiani},
      year={2025},
      eprint={2506.20668},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.20668}, 
}
```

