"""
Implementation of Judged Diffusion Policy with Torque Window and Judgment Weighting
Based on the official robomimic Diffusion Policy implementation.

python train.py --config ../exps/templates/judged_diffusion.json --dataset /home/phi/Documents/SXH/robomimic/datasets/lift/ph/low_dim_v15.hdf5
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

@register_algo_factory_func("judged_diffusion")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the Judged Diffusion algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if algo_config.unet.enabled:
        return JudgedDiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError("Either UNet or Transformer must be enabled for Judged Diffusion Policy")

class JudgedDiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        The ObservationGroupEncoder will process the windowed observations,
        including joint_torque over the last 50 steps (as defined by obs_horizon).
        """
        # Set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # Replace all BatchNorm with GroupNorm to work with EMA
        # Performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)

        # obs_dim reflects the total size after encoding the windowed observations
        # e.g., if obs_horizon=50, joint_pos=(7,), joint_torque=(7,), then their contribution is (50*(7+7))
        obs_dim = obs_encoder.output_shape()[0]

        # Create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            # global_cond_dim now reflects the size after obs_encoder processes the 50-step window including torque
            global_cond_dim=obs_dim
        )

        # The final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)

        # Setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError("Either DDPMScheduler or DDIMScheduler must be enabled")

        # Setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        # Set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        # --- Add judge-related state ---
        self.judgment_weighting_enabled = self.algo_config.judgment_weighting
        self.judgment_weights = self.algo_config.judgment_weights
        self.judgment_threshold = self.algo_config.judgment_success_threshold
        # Example: Store recent episode metrics for simulation purposes
        # In a real implementation, this would come from the environment/replay buffer system
        self.recent_episode_metrics = {} # Maps episode_id -> {success, time, effort}

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        This example assumes batch might contain episode_id information
        if judgment weighting is enabled.
        """
        To = self.algo_config.horizon.observation_horizon # Should be 50
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]} # This now includes joint_torque[:, :50, :]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # Goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        # Check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True

        # --- Add episode_id to batch if judgment is enabled ---
        if self.judgment_weighting_enabled:
            # Assuming your dataset has an 'episode_id' key for each sample
            # This is crucial for linking batch samples to episode metrics
            input_batch["episode_id"] = batch.get("episode_id", None) # Shape (B,)
            if input_batch["episode_id"] is None:
                print("Warning: judgment_weighting is enabled but 'episode_id' not found in batch. "
                      "Cannot apply judgment weights. Proceeding without weighting.")
                self.judgment_weighting_enabled = False # Disable weighting if not available

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.
        Incorporates judgment-based loss weighting if enabled.
        """
        To = self.algo_config.horizon.observation_horizon # Should be 50
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(JudgedDiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]

            # Encode obs - NOW INCLUDES JOINT TORQUE OVER 50 TIMESTEPS
            inputs = {
                "obs": batch["obs"], # This dict now contains joint_torque key with 50-step window
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # First two dimensions should be [B, To] for inputs (e.g., [B, 50])
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            # This call now processes the 50-step window of joint_torque along with other obs
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, To, D_agg] where D_agg includes 50 * torque_dim

            # Flatten obs features for global conditioning (standard for ConditionalUnet1D)
            obs_cond = obs_features.flatten(start_dim=1) # [B, To*D_agg]

            # Sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device) # [B, Tp, Da]

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()

            # Add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps) # [B, Tp, Da]

            # Predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond) # [B, Tp, Da] - obs_cond includes processed 50-step torque

            # --- Calculate Loss with Potential Judgment Weighting ---
            # 1. Standard MSE Loss (unweighted)
            unweighted_loss = F.mse_loss(noise_pred, noise, reduction='none') # [B, Tp, Da]

            # 2. Calculate judgment weights if enabled
            if self.judgment_weighting_enabled:
                # --- Simulate fetching episode metrics based on episode_id ---
                # In a real implementation, you would query a central metrics store
                # or the replay buffer using batch["episode_id"].
                # For simulation, we'll use a dummy mechanism or stored values.
                episode_ids = batch["episode_id"] # [B,] - Shape depends on how it's stored

                # Simulate fetching metrics (replace this logic with your actual metric fetching)
                # Assume self.recent_episode_metrics is populated elsewhere
                weights = torch.ones(B, device=self.device, dtype=torch.float) # Default weight is 1
                for i, ep_id in enumerate(episode_ids):
                    # This is the key lookup: find the performance metrics for episode ep_id
                    metrics = self.recent_episode_metrics.get(ep_id.item(), None) # Example: ep_id is a scalar tensor
                    if metrics is not None:
                        # Calculate a composite score based on your criteria
                        # Example: J = w_s*S + w_t*(1/T_norm) + w_e*(1/E_norm)
                        # where S is success (0/1), T_norm and E_norm are normalized time/effort (0-1 scale)
                        # Weights come from config
                        score = (
                            self.judgment_weights["success"] * metrics.get("success", 0.0) +
                            self.judgment_weights["time"] * (1.0 / max(metrics.get("time", 1.0), 0.1)) + # Inverse time, prevent div by zero
                            self.judgment_weights["effort"] * (1.0 / max(metrics.get("effort", 1.0), 0.1)) # Inverse effort
                        )
                        # Normalize the score or clip it if necessary
                        # For simplicity, we just use the raw score as weight here
                        weights[i] = score # This score acts as the weight

                # Reshape weights to be broadcastable with loss [B, Tp, Da]
                weights = weights.view(B, 1, 1) # [B, 1, 1]

                # Apply weights to the loss
                weighted_loss_per_sample = unweighted_loss.mean(dim=(1,2)) * weights.flatten() # Mean over Tp, Da first, then multiply weight
                loss = weighted_loss_per_sample.mean() # Mean over batch

            else:
                # If judgment weighting is disabled, use standard MSE
                loss = unweighted_loss.mean() # [1]

            # Logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # Gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                # Update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)

                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        """
        log = super(JudgedDiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # Setup inference queues
        To = self.algo_config.horizon.observation_horizon # Should be 50
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do] (includes 50-step torque window)
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D] (e.g., joint_torque: [1, 50, 7])
        To = self.algo_config.horizon.observation_horizon # Should be 50
        Ta = self.algo_config.horizon.action_horizon

        # Ensure obs_queue has the required history
        # This requires the rollout logic to correctly manage obs_queue
        # (e.g., appending current obs, popping oldest if full)
        # The default reset and get_action logic in the official impl might need adjustment
        # to maintain the full 50-step window correctly in obs_queue.
        # For this example, assume obs_queue is managed correctly externally or
        # the obs_dict passed already contains the full windowed history.
        # Standard diffusion policy inference often uses a sliding window internally,
        # but here the encoder expects a fixed To-length sequence.
        # This part might require careful handling depending on rollout implementation.

        if len(self.action_queue) == 0:
            # No actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict)

            # Put actions into the queue
            self.action_queue.extend(action_sequence[0])

        # Has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()

        # [1,Da]
        action = action.unsqueeze(0)
        return action

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon # Should be 50
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError("Either DDPMScheduler or DDIMScheduler must be enabled for inference")

        # Select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        # Encode obs - NOW INCLUDES 50-STEP TORQUE WINDOW
        inputs = {
            "obs": obs_dict, # e.g., joint_torque: [1, 50, 7]
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # First two dimensions should be [B, T] for inputs (e.g., [1, 50])
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # Adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                # However, for obs_horizon=50, this should NOT happen if obs_dict is already windowed correctly.
                # Ensure obs_dict passed in already has the correct shape [1, 50, ...] for each obs key.
                pass # Assume obs_dict is already correctly shaped [1, To, ...]
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, To, D]
        B = obs_features.shape[0]

        # Reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1) # [B, To*D_agg]

        # Initialize action from Gaussian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action

        # Init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction,
                timestep=k,
                global_cond=obs_cond # global_cond now includes processed 50-step torque
            )

            # Inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # Process action using Ta
        start = To - 1 # For obs_horizon=50, this is 49
        end = start + Ta
        action = naction[:,start:end] # [B, Ta, Da]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# --- Add this line to robomimic/algo/__init__.py ---
# from robomimic.algo.judged_diffusion import JudgedDiffusionPolicyUNet