# robomimic/algo/torque_hybrid.py
import torch
import torch.nn as nn
from collections import OrderedDict
from robomimic.algo import register_algo_factory_func
from robomimic.algo.bcq import BCQ  # We inherit structure, but reimplement core
from robomimic.algo.policy_algos import PolicyAlgo
from robomimic.algo.value_algos import ValueAlgo
from robomimic.models.obs_nets import ObsNets
from robomimic.models.policy_nets import ActorNetwork
from robomimic.models.value_nets import ActionValueNetwork
from robomimic.utils.tensor_utils import TorchUtils


@register_algo_factory_func("torque_hybrid")
def algo_config_to_class(algo_config):
    return TorqueHybrid, {}


class TorqueHybrid(PolicyAlgo, ValueAlgo):
    def __init__(self, ...):
        super().__init__(...)
        self.actor_update_counter = 0

    def _create_networks(self):
        self.nets = nn.ModuleDict()
        self._create_critics()
        self._create_actor()

        # Sync target networks
        with torch.no_grad():
            for i in range(len(self.nets["critic"])):
                TorchUtils.hard_update(self.nets["critic"][i], self.nets["critic_target"][i])
            TorchUtils.hard_update(self.nets["actor"], self.nets["actor_target"])
        self.nets = self.nets.float().to(self.device)

    def _create_critics(self):
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            goal_shapes=self.goal_shapes,
            **ObsNets.obs_encoder_args_from_config(self.obs_config.encoder),
        )
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            self.nets["critic"].append(ActionValueNetwork(**critic_args))
            self.nets["critic_target"].append(ActionValueNetwork(**critic_args))

    def _create_actor(self):
        actor_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            goal_shapes=self.goal_shapes,
            **ObsNets.obs_encoder_args_from_config(self.obs_config.encoder),
        )
        self.nets["actor"] = ActorNetwork(**actor_args)
        self.nets["actor_target"] = ActorNetwork(**actor_args)

    def train_on_batch(self, batch, epoch, validate=False):
        info = OrderedDict()
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Train critic
            critic_info = self._train_critic_on_batch(batch, epoch, validate)
            info.update(critic_info)

            # Train actor less frequently
            if not validate:
                self.actor_update_counter += 1
            do_actor_update = (self.actor_update_counter % self.algo_config.actor.update_freq == 0)
            if do_actor_update:
                actor_info = self._train_actor_on_batch(batch, epoch, validate)
                info.update(actor_info)

                # Update targets only on actor update
                with torch.no_grad():
                    for i in range(len(self.nets["critic"])):
                        TorchUtils.soft_update(self.nets["critic"][i], self.nets["critic_target"][i], self.algo_config.target_tau)
                    TorchUtils.soft_update(self.nets["actor"], self.nets["actor_target"], self.algo_config.target_tau)
        return info

    def _train_critic_on_batch(self, batch, epoch, validate=False):
        s, a, r, ns, done = batch["obs"], batch["actions"], batch["rewards"], batch["next_obs"], (1 - batch["dones"])
        q_targets = self._get_target_values(ns, batch.get("goal_obs", None), r, done)

        info = OrderedDict()
        for i, critic in enumerate(self.nets["critic"]):
            q_pred = critic(s, a, batch.get("goal_obs", None))
            loss = nn.MSELoss()(q_pred, q_targets)
            info[f"critic/critic{i+1}_loss"] = loss
            if not validate:
                TorchUtils.backprop_for_loss(critic, self.optimizers["critic"][i], loss)
        return info

    def _get_target_values(self, next_obs, goal_obs, rewards, dones):
        with torch.no_grad():
            next_actions = self.nets["actor_target"](next_obs, goal_obs)
            noise = (torch.randn_like(next_actions) * self.algo_config.actor.noise_std).clamp(
                -self.algo_config.actor.noise_clip, self.algo_config.actor.noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1, 1)

            q1 = self.nets["critic_target"][0](next_obs, next_actions, goal_obs)
            q2 = self.nets["critic_target"][1](next_obs, next_actions, goal_obs)
            q_min = torch.min(q1, q2)
            return rewards + dones * self.algo_config.discount * q_min

    def _train_actor_on_batch(self, batch, epoch, validate=False):
        s = batch["obs"]
        a_demo = batch["actions"]
        goal = batch.get("goal_obs", None)

        pred_actions = self.nets["actor"](s, goal)
        q_val = self.nets["critic"][0](s, pred_actions, goal)

        lam = self.algo_config.alpha / (torch.abs(q_val).mean().detach() + 1e-6)
        actor_loss = -lam * q_val.mean() + nn.MSELoss()(pred_actions, a_demo)

        info = OrderedDict({"actor/loss": actor_loss})
        if not validate:
            TorchUtils.backprop_for_loss(self.nets["actor"], self.optimizers["actor"], actor_loss)
        return info

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        return self.nets["actor"](obs_dict, goal_dict)
