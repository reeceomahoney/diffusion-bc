"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import threading
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import open_dict

def wait_for_enter(env_runner):
    input()
    env_runner.env.kill_server()
    sys.exit(0)

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', default='data/eval_outputs')
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    with open_dict(cfg):
        cfg.task.env.num_envs = 1
        cfg.task.env.server_port = 8081
        cfg.task.env.max_time = 10000
        # cfg.task.env.enable_dynamics_randomization = True
        # cfg.policy.x_sampling_steps = 4


    cls = hydra.utils.get_class(cfg._target_)
    # workspace = cls(cfg, output_dir=output_dir)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        env_cfg=cfg.task.env,
        obs_dim=cfg.task.obs_dim,)
    
    threading.Thread(target=wait_for_enter, args=(env_runner,)).start()
    runner_log = env_runner.run(policy, eval_run=True)
    
    # dump log to json
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # out_path = os.path.join(output_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
