from typing import Any, Dict, Optional, Sequence, List, cast
import torch
from torch import Tensor

from d3rlpy.dataset import TransitionMiniBatch, Transition
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.algos.cql import CQL
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR
from d3rlpy.models.encoders import EncoderFactory

from myd3rlpy.algos.st import STBase


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STCQL(STBase, CQL):
    r"""Twin Delayed Deep Deterministic Policy Gradients algorithm.
    TD3 is an improved DDPG-based algorithm.
    Major differences from DDPG are as follows.
    * TD3 has twin Q functions to reduce overestimation bias at TD learning.
      The number of Q functions can be designated by `n_critics`.
    * TD3 adds noise to target value estimation to avoid overfitting with the
      deterministic policy.
    * TD3 updates the policy function after several Q function updates in order
      to reduce variance of action-value estimation. The interval of the policy
      function update can be designated by `update_actor_interval`.
    .. math::
        L(\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \min_j Q_{\theta_j'}(s_{t+1}, \pi_{\phi'}(s_{t+1}) +
            \epsilon) - Q_{\theta_i}(s_t, a_t))^2]
    .. math::
        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\min_i Q_{\theta_i}(s_t, \pi_\phi(s_t))]
    where :math:`\epsilon \sim clip (N(0, \sigma), -c, c)`
    References:
        * `Fujimoto et al., Addressing Function Approximation Error in
          Actor-Critic Methods. <https://arxiv.org/abs/1802.09477>`_
    Args:
        actor_learning_rate (float): learning rate for a policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.td3_impl.TD3Impl): algorithm implementation.
    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    # actor必须被重放，没用选择。
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    # _dynamics: Optional[ProbabilisticEnsembleDynamics]
    _rollout_interval: int
    _rollout_horizon: int
    _rollout_batch_size: int
    _use_gpu: Optional[Device]
    _critic_replay_type: bool
    _critic_replay_lambda: float
    _actor_replay_type: bool
    _actor_replay_lambda: float
    _replay_model: bool
    _generate_step: int
    _select_time: int
    _model_noise: float

    _task_id: str
    _single_head: bool
    _merge: bool

    def __init__(
        self,
        critic_replay_type='bc',
        critic_replay_lambda=1,
        actor_replay_type='rl',
        actor_replay_lambda=1,
        gem_alpha: float = 1,
        agem_alpha: float = 1,
        ewc_rwalk_alpha: float = 0.5,
        damping: float = 0.1,
        epsilon: float = 0.1,
        impl_name = 'co',
        # n_train_dynamics = 1,
        retrain_topk = 4,
        log_prob_topk = 10,
        experience_type = 'random_transition',
        sample_type = 'retrain',
        match_prop_quantile = 0.5,
        match_epsilon = 0.1,
        random_sample_times = 10,

        critic_update_step = 0,

        clone_critic = False,
        clone_actor = False,
        merge = False,
        coldstart_step = 0,

        fine_tuned_step = 1,
        std_time = 1,
        std_type = 'clamp',

        **kwargs: Any
    ):
        super(STBase, self).__init__(**kwargs)
        self._critic_replay_type = critic_replay_type
        self._critic_replay_lambda = critic_replay_lambda
        self._actor_replay_type = actor_replay_type
        self._actor_replay_lambda = actor_replay_lambda

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._damping = damping
        self._epsilon = epsilon

        self._impl_name = impl_name
        # self._n_train_dynamics = n_train_dynamics
        self._retrain_topk = retrain_topk
        self._log_prob_topk = log_prob_topk
        self._experience_type = experience_type
        self._sample_type = sample_type

        self._begin_grad_step = 0

        self._match_prop_quantile = match_prop_quantile
        self._match_epsilon = match_epsilon
        self._random_sample_times = random_sample_times

        self._critic_update_step = critic_update_step

        self._clone_critic = clone_critic
        self._clone_actor = clone_actor
        self._coldstart_step = coldstart_step
        self._merge = merge
        self._fine_tuned_step = fine_tuned_step
        self._std_time = std_time
        self._std_type = std_type

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int) -> None:
        impl_dict = {
            'observation_shape':observation_shape,
            'action_size':action_size,
            'actor_learning_rate':self._actor_learning_rate,
            'critic_learning_rate':self._critic_learning_rate,
            'temp_learning_rate':self._temp_learning_rate,
            'alpha_learning_rate':self._alpha_learning_rate,
            'actor_optim_factory':self._actor_optim_factory,
            'critic_optim_factory':self._critic_optim_factory,
            'temp_optim_factory':self._temp_optim_factory,
            'alpha_optim_factory':self._alpha_optim_factory,
            'actor_encoder_factory':self._actor_encoder_factory,
            'critic_encoder_factory':self._critic_encoder_factory,
            'q_func_factory':self._q_func_factory,
            'critic_replay_type':self._critic_replay_type,
            'critic_replay_lambda':self._critic_replay_lambda,
            'actor_replay_type':self._actor_replay_type,
            'actor_replay_lambda':self._actor_replay_lambda,
            'gamma':self._gamma,
            'gem_alpha':self._gem_alpha,
            'agem_alpha':self._agem_alpha,
            'ewc_rwalk_alpha':self._ewc_rwalk_alpha,
            'damping':self._damping,
            'epsilon':self._epsilon,
            'tau':self._tau,
            'n_critics':self._n_critics,
            'conservative_weight': self._conservative_weight,
            'initial_temperature':self._initial_temperature,
            'initial_alpha':self._initial_alpha,
            'alpha_threshold':self._alpha_threshold,
            'n_action_samples':self._n_action_samples,
            'soft_q_backup':self._soft_q_backup,
            'use_gpu':self._use_gpu,
            'scaler':self._scaler,
            'action_scaler':self._action_scaler,
            'reward_scaler':self._reward_scaler,
            'fine_tuned_step': self._fine_tuned_step,
        }
        if self._impl_name == 'cql':
            from myd3rlpy.algos.torch.st_cql_impl import STCQLImpl as STImpl
            impl_dict["std_time"] = self._std_time
            impl_dict["std_type"] = self._std_type
        # elif self._impl_name in ['mgcql', 'mqcql', 'mrcql']:
        #     if self._impl_name == 'mgcql':
        #         from myd3rlpy.algos.torch.st_mgcql_impl import STImpl as STImpl
        #     if self._impl_name == 'mqcql':
        #         from myd3rlpy.algos.torch.st_mqcql_impl import STImpl as STImpl
        #     elif self._impl_name == 'mrcql':
        #         from myd3rlpy.algos.torch.st_mrcql_impl import STImpl as STImpl
        #         impl_dict['random_sample_times'] = self._random_sample_times
        #     impl_dict['match_prop_quantile'] = self._match_prop_quantile
        #     impl_dict['match_epsilon'] = self._match_epsilon
        else:
            print(self._impl_name)
            raise NotImplementedError
        self._impl = STImpl(
            **impl_dict
        )
        self._impl.build()

    # 注意欧氏距离最近邻被塞到actions后面了。
    def _update(self, batch: TransitionMiniBatch, online: bool, batch_num: int, total_step: int, coldstart_step: Optional[int] = None, replay_batch: Optional[List[Tensor]]=None) -> Dict[int, float]:
        if coldstart_step is None:
            coldstart_step = self._coldstart_step
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        metrics = {}
        if not self._merge or total_step < coldstart_step:
            if self._temp_learning_rate > 0:
                #if total_step % 1000 == 0:
                #    h = self._impl._policy._encoder(torch.from_numpy(batch.observations).to(self._impl.device))
                #    mu = self._impl._policy._mu(h)
                #    logstd = cast(torch.nn.Linear, self._impl._policy._logstd)(h)
                #    clipped_logstd = logstd.clamp(self._impl._policy._min_logstd, self._impl._policy._max_logstd)
                #try:
                temp_loss, temp = self._impl.update_temp(batch)
                #except ValueError:
                #    h = self._impl._policy._encoder(torch.from_numpy(batch.observations).to(self._impl.device))
                #    mu = self._impl._policy._mu(h)
                #    logstd = cast(torch.nn.Linear, self._impl._policy._logstd)(h)
                #    clipped_logstd = logstd.clamp(self._impl._policy._min_logstd, self._impl._policy._max_logstd)
                #    policy_sample, policy_logstd = self._impl._policy.sample_with_log_prob(torch.from_numpy(batch.observations).to(self._impl.device))
                metrics.update({"temp_loss": temp_loss, "temp": temp})
            if self._alpha_learning_rate > 0:
                alpha_loss, alpha = self._impl.update_alpha(batch)
                metrics.update({"alpha_loss": alpha_loss, "alpha": alpha})

            critic_loss, replay_critic_loss = self._impl.update_critic(batch, replay_batch, clone_critic=self._clone_critic, online=online)
            metrics.update({"critic_loss": critic_loss})
            metrics.update({"replay_critic_loss": replay_critic_loss})

            if (total_step > self._critic_update_step and total_step < coldstart_step) or self._impl._impl_id == 0:
                actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, clone_actor=self._clone_actor, online=online)
                # actor_loss, replay_actor_loss = self._impl.update_actor(batch, replay_batch, online=online)
                metrics.update({"actor_loss": actor_loss})
                metrics.update({"replay_actor_loss": replay_actor_loss})

            self._impl.update_critic_target()
            self._impl.update_actor_target()
        elif not online:
            self._merge_update(batch, replay_batch)

        return metrics

    def generate_new_data(
        self, transitions: List[Transition], real_observation_size, real_action_size, batch_size = 64,
    ) -> Optional[List[Transition]]:
        return None
