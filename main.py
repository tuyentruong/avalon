#!/usr/bin/env python
"""
Playground for trying out Generally Intelligent's Avalon (https://generallyintelligent.ai/avalon/).
"""
import os
import pathlib
import shutil
import subprocess
from datetime import datetime

import torch
from attr import evolve
from gym.utils.env_checker import check_env
from tree import map_structure

from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.trainer import OnPolicyTrainer
from avalon.agent.common.types import StepData
from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.agent.train_ppo_avalon import AvalonPPOParams
from avalon.common.log_utils import configure_local_logger
from avalon.common.visual_utils import encode_video
from avalon.datagen.env_helper import observation_video_array

from avalon.datagen.godot_env.interactive_godot_process import GODOT_ERROR_LOG_PATH, GODOT_BINARY_PATH
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.world_generator import generate_world, GenerateAvalonWorldParams


def get_step_data(config, actions, received_obs, reward, done, info):
    received_obs = {k: torch.from_numpy(v) for k, v in received_obs.items()}

    # is_terminal indicates if a true environment termination happened (not a time limit)
    is_terminal = done and not info.get("TimeLimit.truncated", False)
    assert isinstance(reward, (int, float))
    stored_obs = {}
    for k, v in received_obs.items():
        if config.trainer.params.obs_first:
            # Since `received_obs` comes after the reward/done, we need to use the obs from last step
            stored_obs[k] = config.trainer.train_rollout_manager.workers[0].next_obs[k]
            config.trainer.train_rollout_manager.workers[0].next_obs[k] = v
        else:
            if done:
                # We need to handle this case differently. received_obs is actually from the next ep;
                # it wouldn't be good to store that as part of this one.
                # Instead, we store the "terminal observation" which is the one received along with the done signal.
                # We handle both the time_limit and no_time_limit case the same.
                stored_obs[k] = torch.from_numpy(info["terminal_observation"][k])
            else:
                # Not done
                stored_obs[k] = v

    step_data = StepData(
        observation=stored_obs,
        reward=reward,
        done=done,
        is_terminal=is_terminal,
        info=info,
        action=actions,
    )
    # We return the unmodified `received_obs` here to be used for the next inference pass.
    # At a reset, this will be the new observation from the next episode, which is indeed what we want for inference.
    return received_obs, step_data


def env_step(config, env, i):
    if config.use_ppo_trainer:
        step_actions, to_store = config.trainer.train_rollout_manager.run_inference(config.exploration_mode)
        action = map_structure(lambda x: x[0].numpy(), step_actions)
        obs, reward, done, info = env.step(action)
        received_obs, step_data = get_step_data(config, step_actions, obs, reward, done, info)
        step_data = config.trainer.train_rollout_manager.model.build_algorithm_step_data(step_data, extra_info=map_structure(lambda x: x[0], to_store))
        if done and config.trainer.params.time_limit_bootstrapping and step_data.info.get("TimeLimit.truncated", False) is True:
            info["terminal_observation"] = obs
            step_data = evolve(step_data, reward=step_data.reward + config.trainer.train_rollout_manager.time_limit_bootstrapping(info))
        config.trainer.train_rollout_manager.dones[0] = done
        step_data_batch = {}
        step_data_batch[0] = step_data
        config.trainer.train_rollout_manager.storage.add_timestep_samples(step_data_batch)
    else:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    task = info["task"]
    action = action["discrete"]
    hit_points = info["hit_points"]
    score = info["score"]
    success = info["success"]
    remaining_frames = info["remaining_frames"]
    print("[{0}] task = {1}, action = {2}, reward = {3}, hit_points = {4}, score = {5}, remaining frames = {6}, success = {7}".format(i, task, str(action).replace("\n", ""), reward, hit_points, score, remaining_frames, success))
    if done or remaining_frames == 0:
        print("{0} task done: success = {1}, remaining frames = {2}".format(info["task"], success, remaining_frames))
        env.reset()
    return obs, success


class Config():
    use_ppo_trainer: bool
    exploration_mode: str
    check_env: bool
    load_world_manually: bool
    num_steps: int
    fps: int
    stop_when_success: bool
    display_video: bool

    def __init__(self):
        self.use_ppo_trainer = False
        self.exploration_mode = "explore"
        self.check_env = False # whether to validate that the Avalon gym environment conforms to the gym interface
        self.load_world_manually = False
        self.num_steps = 30_000
        self.fps = 10
        self.stop_when_success = False
        self.display_video = False


def main():
    configure_local_logger()
    print("GODOT_BINARY_PATH = {0}".format(GODOT_BINARY_PATH))
    print("GODOT_ERROR_LOG_PATH = {0}".format(GODOT_ERROR_LOG_PATH))

    config = Config()

    env_params = GodotEnvironmentParams(
        #mode="val",
        resolution=1024,
        training_protocol=TrainingProtocolChoice.MULTI_TASK_BASIC,
        initial_difficulty=1,
        is_debugging_godot=True,
        is_logging_artifacts_on_error_to_s3=False,
    )
    if config.use_ppo_trainer:
        params = parse_args(AvalonPPOParams())
        config.trainer = OnPolicyTrainer(params)
        config.trainer.train_rollout_manager.workers[0].worker.lazy_init_env()
        env = config.trainer.train_rollout_manager.workers[0].worker.env
    else:
        env = AvalonEnv(env_params)
    if config.check_env:
        check_env(env, skip_render_check=False)
        return
    if config.load_world_manually:
        OUTPUT_FOLDER = pathlib.Path("./outputs/").absolute()
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        params = generate_world(
            GenerateAvalonWorldParams(
                AvalonTask.MOVE,
                difficulty=1,
                seed=42,
                index=0,
                output=str(OUTPUT_FOLDER),
            )
        )
        env.reset_nicely_with_specific_world(episode_seed=0, world_params=params)
    else:
        env.reset()
    print("Getting observations...")
    observations = []
    for i in range(config.num_steps):
        obs, success = env_step(config, env, i)
        observations.append(obs)
        if config.stop_when_success and success:
            break
    if config.display_video:
        print("Generating video from observations...")
        frames = observation_video_array(observations)
        video_path = encode_video(frames, normalize=False, fps=config.fps, video_format="webm")
        output_path = os.getcwd() + "/outputs/{0}.webm".format(datetime.now().strftime("%Y%m%d%H%M"))
        shutil.copyfile(video_path, output_path)
        print("Video saved to {0}".format(output_path))
        subprocess.Popen(["open", output_path])


if __name__ == '__main__':
    main()
