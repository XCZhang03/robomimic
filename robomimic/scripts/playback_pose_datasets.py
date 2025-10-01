"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize depth observations along with image observations
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --render_depth_names agentview_depth \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import random

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.vis_utils import depth_to_rgb
from robomimic.envs.env_base import EnvBase, EnvType

from pathlib import Path
from PIL import Image


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


class CameraPoseWriter:
    """
    A simple writer class to save poses over time as numpy arrays.
    Works like video_writer but for poses.
    """
    def __init__(self, save_path, camera_names):
        """
        Args:
            save_path (str): file path prefix for saving poses (without extension).
            camera_names (list): list of camera names whose poses will be saved.
        """
        self.save_path = save_path
        self.camera_names = camera_names
        self.poses = {cam: [] for cam in camera_names}
        self.timesteps = []

    def append_data(self, pose_dict, timestep):
        """
        Save poses for this timestep.

        Args:
            pose_dict (dict): mapping {camera_name: pose_array}
            timestep (int): the current timestep
        """
        self.timesteps.append(timestep)
        for cam, pose in pose_dict.items():
            if cam in self.poses:
                self.poses[cam].append(pose)

    def close(self):
        """
        Dump all collected poses to disk as npz (one array per camera).
        """
        poses_np = {cam: np.array(self.poses[cam]) for cam in self.camera_names}
        poses_np["timesteps"] = np.array(self.timesteps)
        np.savez(os.path.join(self.save_path, "camera_poses.npz"), **poses_np)


def noise_fn(states: np.ndarray, actions: np.ndarray, noise: float, env, gripper_only=False, min_len: int=40):
    if noise == 0:
        return states, actions
    traj_len = actions.shape[0]
    if min_len is None:
        min_len = traj_len
    noising_start=np.random.choice(traj_len - min_len) if traj_len > min_len else 0 # at least 8 frames (40 steps)
    if actions.shape[-1] == 14:
        gripper_index = [6, 13]
    elif actions.shape[-1] == 7:
        gripper_index = [6,]
    else:
        gripper_index = []
    if len(gripper_index) > 0:
        actions[noising_start:, gripper_index] = actions[noising_start:, gripper_index] * ((np.random.rand(traj_len - noising_start, len(gripper_index)) > noise).astype(np.float32) * 2 - 1)
    gaussian_noise = np.random.normal(0, noise, actions.shape) if not gripper_only else np.zeros_like(actions)
    actions[noising_start:] += gaussian_noise[noising_start:]
    action_low, action_high = env.env.action_spec
    actions = np.clip(actions, action_low, action_high)
    actions = actions[noising_start:]
    states = states[noising_start:]
    return states, actions

def split_trajectory(args, states: np.ndarray, actions: np.ndarray):
    traj_len = args.traj_len
    splits = []
    start = 0
    while start < states.shape[0]:
        end = min(start + traj_len, states.shape[0])
        if (args.eval and (end - start) < traj_len) or (not args.eval and (end - start) < 40):
            break ## only retain full traj_len segments for eval
        splits.append((states[start:end], actions[start:end]))
        start += random.randint(20, 40) if not args.eval else (traj_len // 2) # 40 steps overlap, 8 frames
    return splits

def playback_trajectory_with_env(
    env, 
    empty_env,
    initial_state, 
    states, 
    actions=None, 
    render=False, 
    video_writers=None, 
    pose_video_writers=None,
    camera_pose_writer=None,
    video_skip=5, 
    action_chunk=None,
    camera_names=None,
    first=False,
    res=128,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)
    write_video = (video_writers is not None)
    video_count = 0
    assert not (render and write_video)
    if action_chunk is None:
        action_chunk = video_skip

    unwrapped_empty_env = empty_env.env
    unwrapped_env = env.base_env
    unwrapped_empty_env.copy_env_model(unwrapped_env)
    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    env.reset()
    env.reset_to(initial_state)
    unwrapped_empty_env.copy_robot_state(unwrapped_env)

    # update camera_names for available cameras
    camera_names = list(set(unwrapped_env.sim.model.camera_names).intersection(unwrapped_empty_env.sim.model.camera_names).intersection(set(camera_names)).intersection(set(video_writers.keys())))
    if len(camera_names) == 0:
        return
    
    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in range(traj_len):
        if action_playback:
            env.step(actions[i])
            unwrapped_empty_env.step(actions[i])
            # if i < traj_len - 1:
            #     # check whether the actions deterministically lead to the same recorded states
            #     state_playback = env.get_state()["states"]
            #     if not np.all(np.equal(states[i + 1], state_playback)):
            #         err = np.linalg.norm(states[i + 1] - state_playback)
            #         print("warning: playback diverged by {} at step {}".format(err, i))
        else:
            env.reset_to({"states" : states[i]})
            empty_env.copy_robot_state(env)

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                for cam_name in camera_names:
                    video_img = env.render(mode="rgb_array", height=res, width=res, camera_name=cam_name)
                    video_writers[cam_name].append_data(video_img)
                    if 'robot' not in cam_name:
                        cam_transform = unwrapped_empty_env.get_camera_transform(camera_name=cam_name, camera_height=res, camera_width=res)
                        pose_image = unwrapped_empty_env.plot_pose(cam_transform, height=res, width=res)
                        pose_video_writers[cam_name].append_data(pose_image)
                camera_pose_writer.append_data(unwrapped_empty_env.get_camera_pose(), video_count)
            video_count += 1
            if video_count % action_chunk == 0:
                unwrapped_empty_env.copy_robot_state(unwrapped_env)

        if first:
            break


def playback_trajectory_with_obs(
    traj_grp,
    video_writer, 
    video_skip=5, 
    image_names=None,
    depth_names=None,
    first=False,
):
    """
    This function reads all "rgb" (and possibly "depth") observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        depth_names (list): determines which depth observations are used for rendering (if any).
        first (bool): if True, only use the first frame of each episode.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    if depth_names is not None:
        # compute min and max depth value across trajectory for normalization
        depth_min = { k : traj_grp["obs/{}".format(k)][:].min() for k in depth_names }
        depth_max = { k : traj_grp["obs/{}".format(k)][:].max() for k in depth_names }

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            depth = [depth_to_rgb(traj_grp["obs/{}".format(k)][i], depth_min=depth_min[k], depth_max=depth_max[k]) for k in depth_names] if depth_names is not None else []
            frame = np.concatenate(im + depth, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def playback_dataset(args):
    # some arg checking
    if args.video_path is None:
        # args.video_path = os.path.dirname(args.dataset)
        # Find the relative path of args.dataset to ./dataset
        dataset_dir = Path("./datasets")
        video_dir = str(dataset_dir) + f"_std_{args.noise}" + f"_{args.res}" + f"_chunk{args.action_chunk}" + (f"_gripper" if args.gripper_only else "") + (f"_len{args.traj_len}" if args.traj_len is not None else "") \
            + (f"_eval" if args.eval else "")
        rel_dataset_path = os.path.relpath(args.dataset, str(dataset_dir))
        print(f"Relative dataset path: {rel_dataset_path}")
        args.video_path = os.path.join(video_dir, rel_dataset_path)
        os.makedirs(args.video_path, exist_ok=True)
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    if args.render_depth_names is not None:
        assert args.use_obs, "depth observations can only be visualized from observations currently"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)
        empty_env = EnvUtils.create_empty_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)
        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    
    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[:args.n]

    demos = demos[222:223]
    

    for ind in range(len(demos)):
        ep = demos[ind]
        video_dir = os.path.join(args.video_path, ep)
        os.makedirs(video_dir, exist_ok=True)

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                depth_names=args.render_depth_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        # supply actions if using open-loop action playback
        # actions = None
        # if args.use_actions:
        # always use actions
        actions = f["data/{}/actions".format(ep)][()]
        # Dump states and actions for this episode
        np.save(os.path.join(video_dir, "states.npy"), states)
        print("ep len:", states.shape[0])
        if actions is not None:
            np.save(os.path.join(video_dir, "actions.npy"), actions)
        with open(os.path.join(video_dir, "args.json"), "w") as f_args:
            json.dump(vars(args), f_args, indent=4)

        for i, (seg_states, seg_actions) in enumerate(split_trajectory(args, states, actions)):

            # maybe dump video
            if write_video:
                video_writers = {}
                pose_video_writers = {}
                for camera_name in args.render_image_names:
                    camera_video_path = os.path.join(video_dir, f"{camera_name}_seg{i}.mp4")
                    if os.path.exists(camera_video_path):
                        # print(f"skipping camera {camera_name}", flush=True)
                        continue
                    video_writers[camera_name] = imageio.get_writer(camera_video_path, fps=20)
                    if 'robot' not in camera_name:
                        pose_video_path = os.path.join(video_dir, f"{camera_name}_seg{i}_pose.mp4")
                        pose_video_writers[camera_name] = imageio.get_writer(pose_video_path, fps=20)
                camera_pose_writer = CameraPoseWriter(video_dir, empty_env.env.sim.model.camera_names)
            
            camera_names = list(set(env.base_env.sim.model.camera_names).intersection(empty_env.env.sim.model.camera_names).intersection(set(args.render_image_names)).intersection(set(video_writers.keys())))
            if len(camera_names) == 0:
                continue

            noised_states, noised_actions = noise_fn(seg_states, seg_actions, args.noise, env, gripper_only=args.gripper_only, min_len=None)

            print(f"Playing back seg{i} of episode: {ep} of env {rel_dataset_path} with length {len(noised_states)}", flush=True)

            initial_state = dict(states=noised_states[0])
            if is_robosuite_env:
                # initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
                initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)


            playback_trajectory_with_env(
                env=env, 
                empty_env=empty_env,
                initial_state=initial_state, 
                states=noised_states, actions=noised_actions, 
                render=args.render, 
                video_writers=video_writers, 
                pose_video_writers=pose_video_writers,
                camera_pose_writer=camera_pose_writer,
                video_skip=args.video_skip,
                camera_names=camera_names,
                first=args.first,
                res=args.res,
                action_chunk=args.action_chunk,
            )
            if write_video:
                for video_writer in video_writers.values():
                    video_writer.close()
                for pose_video_writer in pose_video_writers.values():
                    pose_video_writer.close()
                camera_pose_writer.close()
            
    
    f.close()
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=['agentview', 'frontview', 'sideview', 'birdview'],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # depth observations to use for writing to video
    parser.add_argument(
        "--render_depth_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) depth observation(s) to use for rendering to video"
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="(optional) noise level to add to actions"
    )
    parser.add_argument(
        "--res",
        type=int,
        default=128,
        help="The resolution of created videos"
    )
    parser.add_argument(
        "--action-chunk",
        type=int,
        default=None,
        help="(optional) number of steps between copying robot state from env to empty_env during"
    )
    parser.add_argument(
        "--gripper-only",
        action='store_true',
        help="if true, only add noise to gripper actions"
    )
    parser.add_argument(
        "--traj-len",
        type=int,
        default=None,
        help="(optional) if provided, only playback random traj_len steps of each trajectory"
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        help="if true, generate eval videos, maximizing length"
    )

    args = parser.parse_args()
    if args.action_chunk is None:
        args.action_chunk = args.video_skip
    if args.dataset == 'all':
        import threading
        import multiprocessing as mp
        from robomimic.scripts.download_datasets import DATASET_REGISTRY
        default_base_dir = os.path.join(robomimic.__path__[0], "../datasets")
        tasks = []
        for task in DATASET_REGISTRY:
            for dataset_type in DATASET_REGISTRY[task]:
                download_dir = os.path.abspath(os.path.join(default_base_dir, task, dataset_type))
                if not os.path.exists(download_dir):
                    continue
                for file in os.listdir(download_dir):
                    if file.endswith(".hdf5"):
                        from copy import deepcopy
                        cur_args = deepcopy(args)
                        cur_args.dataset = os.path.join(download_dir, file)
                        cur_args.video_path = None
                        tasks.append(cur_args)

        def worker(task_args):
            playback_dataset(task_args)

        processes = []
        for task_args in tasks:
            p = mp.Process(target=worker, args=(task_args,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        playback_dataset(args)
