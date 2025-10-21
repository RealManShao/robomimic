# robomimic/config/torque_hybrid_config.py
from robomimic.config.base_config import BaseConfig

class TorqueHybridConfig(BaseConfig):
    ALGO_NAME = "torque_hybrid"

    def algo_config(self):
        # Optimization
        self.algo.optim_params.critic.learning_rate.initial = 3e-4
        self.algo.optim_params.actor.learning_rate.initial = 3e-4

        # Core TD3-BC parameters
        self.algo.alpha = 2.5          # BC vs RL trade-off
        self.algo.discount = 0.99
        self.algo.target_tau = 0.005
        self.algo.n_step = 1

        # Critic
        self.algo.critic.ensemble.n = 2
        self.algo.critic.layer_dims = (256, 256, 256)
        self.algo.critic.use_huber = False

        # Actor
        self.algo.actor.update_freq = 2
        self.algo.actor.noise_std = 0.2
        self.algo.actor.noise_clip = 0.5
        self.algo.actor.layer_dims = (256, 256, 256)

        # extensions (optional placeholders)
        self.algo.use_torque_in_obs = True      # if torque is part of observation
        self.algo.judgment_weighting = True     # enable judgment-based loss scaling
