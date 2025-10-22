# robomimic/config/judged_diffusion_config.py
from robomimic.config.base_config import BaseConfig

class JudgedDiffusionConfig(BaseConfig):
    ALGO_NAME = "judged_diffusion"

    def algo_config(self):
        # --- Diffusion Policy Specific (Based on Official Implementation) ---
        # This defines the observation horizon (window size for all obs, including torque)
        self.algo.horizon.observation_horizon = 50 # Crucial: Set to 50 for torque window
        self.algo.horizon.action_horizon = 8      # Number of actions to take from generated sequence
        self.algo.horizon.prediction_horizon = 16 # Number of actions to generate in total

        # Network architecture (UNet)
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 128
        self.algo.unet.down_dims = (256, 512, 1024)
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        self.algo.unet.cond_predict_scale = True
        self.algo.unet.use_film_scale = True
        self.algo.unet.use_film_shift = True
        self.algo.unet.use_conv = True
        self.algo.unet.use_group_norm = True
        self.algo.unet.use_global_cond = True
        self.algo.unet.use_prefix_cond = False

        # Scheduler (DDPM)
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.beta_schedule = "squaredcos_cap_v2"
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = "epsilon"

        # EMA (Exponential Moving Average)
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # --- Judgement Specific (Added) ---
        # Flag to enable judgment-based loss weighting
        self.algo.judgment_weighting = True
        # Weights for the judgment components (success, time, effort)
        self.algo.judgment_weights = {
            "success": 1.0,
            "time": 0.5,
            "effort": 0.2 # Effort is typically inverse (lower is better)
        }
        # Threshold for considering an episode "good" for potential masking (optional)
        self.algo.judgment_success_threshold = 0.5

        # --- Optimization ---
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.9
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [50, 100]
        self.algo.optim_params.policy.regularization.L2 = 1e-6

# --- Add this line to robomimic/config/__init__.py ---
# from robomimic.config.judged_diffusion_config import JudgedDiffusionConfig