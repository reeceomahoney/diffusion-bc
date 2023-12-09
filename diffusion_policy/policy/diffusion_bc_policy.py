from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.diffusion_bc_transformer import DiffusionBCTransformer

class DiffusionBCPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: DiffusionBCTransformer,
            noise_scheduler: DDPMScheduler,
            action_dim, 
            num_inference_steps=None,
            x_sampling_steps=0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.x_sampling_steps = x_sampling_steps
    
    # ========= inference  ============
    def conditional_sample(self, nobs, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        action = torch.randn(
            size=(nobs.shape[0], 1, self.action_dim),
            dtype=nobs.dtype,
            device=nobs.device,
            generator=None)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # predict model output
            model_output = model(nobs, action, t)

            # compute previous image: x_t -> x_t-1
            action = scheduler.step(
                model_output, t, action, 
                generator=None,
                **kwargs
                ).prev_sample
        
        for _ in range(self.x_sampling_steps):
            # predict model output
            xt = torch.tensor([1], device=nobs.device, dtype=torch.long)
            model_output = model(nobs, action, xt)

            # compute previous image: x_t -> x_t-1
            action = scheduler.step(
                model_output, 1, action, 
                generator=None,
                **kwargs
                ).prev_sample
        
        return action

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])

        # run sampling
        nsample = self.conditional_sample(nobs, **self.kwargs)
        
        # unnormalize prediction
        action = self.normalizer['action'].unnormalize(nsample)
        
        return { 'action': action, }

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        # sampler returns 2 actions so remove the first one
        action = nbatch['action'][:, 1:]

        # Sample noise that we'll add to the images
        noise = torch.randn(action.shape, device=action.device)
        bsz = action.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=action.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            action, noise, timesteps)
        
        # Predict the noise residual
        pred = self.model(obs, noisy_action, timesteps)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
