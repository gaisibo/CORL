from typing import Dict, Optional, List, Callable, Any, Union, Tuple
from tqdm.auto import trange
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from tqdm.auto import trange
import numpy as np
import torch

from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.online.buffers import Buffer, ReplayBuffer
from d3rlpy.iterators import TransitionIterator
from d3rlpy.base import LearnableBase
from d3rlpy.logger import LOG, D3RLPyLogger
import gym
from myd3rlpy.algos.st import STBase
from myd3rlpy.algos.torch.st_impl import STImpl
from d3rlpy.dataset import TransitionMiniBatch as OldTransitionMiniBatch
from myd3rlpy.dataset import MDPDataset, Episode, TransitionMiniBatch


class O2OBase(STBase):
    def update(self, policy_batch: TransitionMiniBatch, value_batch: TransitionMiniBatch, online: bool = False) -> Dict[int, float]:
    # def update(self, batch: TransitionMiniBatch, online: bool = False, batch_num: int=0, total_step: int=0, replay_batch: Optional[List[Tensor]]=None) -> Dict[int, float]:
        """Update parameters with mini-batch of data.
        Args:
            batch: mini-batch data.
        Returns:
            dictionary of metrics.
        """
        loss = self._update(policy_batch, value_batch, online)
        self._grad_step += 1
        return loss

    def fitter(
        self,
        dataset: Optional[Union[List[Episode], MDPDataset]] = None,
        iterator: Optional[TransitionIterator] = None,
        old_iterator: Optional[TransitionIterator] = None,
        continual_type: str = "er",
        buffer_mix_ratio: float = 0.5,
        buffer_mix_type: str = "all",
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 1000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        scorers_list: Optional[
                List[Dict[str, Callable[[Any, List[Episode]], float]]]
            ] = None,
        eval_episodes_list: Optional[Dict[int, List[Episode]]] = None,
        save_epochs: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
        test: bool = False,
        epoch_num: Optional[int] = None,
    ):
        """Iterate over epochs steps to train with the given dataset. At each
             iteration algo methods and properties can be changed or queried.
        .. code-block:: python
            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)
        Args:
            dataset: list of episodes to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            scorers: list of scorer functions used with `eval_episodes`.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        Returns:
            iterator yielding current epoch and metrics dict.
        """
        if self._impl is None:
            LOG.debug("Building models...")
            observation_shape = dataset.get_observation_shape()
            self.create_impl(
                self._process_observation_shape(observation_shape),
                dataset.get_action_size(),
            )
            self._impl._impl_id = 0
            LOG.debug("Models have been built.")
        else:
            self._impl._impl_id = 0
            # self._impl.rebuild_critic()
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard_dir,
        )

        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        total_step = 0
        print(f'train policy')
        if old_iterator is not None:
            old_iterator.reset()
        for epoch in range(1, n_epochs + 1):
            if epoch > 1 and test:
                break

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(iterator)),
                disable=not show_progress,
                desc=f"Epoch {epoch}/{n_epochs}",
            )

            iterator.reset()
            if old_iterator is not None:
                old_iterator.reset()

            for batch_num, itr in enumerate(range_gen):
                if batch_num > 10 and test:
                    break

                # new_transitions = self.generate_new_data(transitions=iterator.transitions)
                # if new_transitions:
                #     iterator.add_generated_transitions(new_transitions)
                #     LOG.debug(
                #         f"{len(new_transitions)} transitions are generated.",
                #         real_transitions=len(iterator.transitions),
                #         fake_transitions=len(iterator.generated_transitions),
                #     )
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        if old_iterator is not None and continual_type == "er":
                            new_batch = next(iterator)
                            try:
                                old_batch = next(old_iterator)
                            except StopIteration:
                                old_iterator.reset()
                                old_batch = next(old_iterator)
                            part_new_batch = TransitionMiniBatch(new_batch.transitions[:round((1 - buffer_mix_ratio) * self._batch_size)])
                            mix_batch = TransitionMiniBatch(
                                    part_new_batch.transitions + old_batch.transitions
                                    )
                            if buffer_mix_type == 'all':
                                value_batch = policy_batch = mix_batch
                            elif buffer_mix_type == 'policy':
                                policy_batch = mix_batch
                                value_batch = new_batch
                            elif buffer_mix_type == 'value':
                                policy_batch = new_batch
                                value_batch = mix_batch
                            else:
                                raise NotImplementedError
                        else:
                            policy_batch = next(iterator)
                            value_batch = policy_batch

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(policy_batch, value_batch, online=False)
                        # self._impl.increase_siamese_alpha(epoch - n_epochs, itr / len(iterator))

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    try:
                        # logger.add_metric("weight", self._impl._weight)
                        # logger.add_metric("log_probs", self._impl._log_probs)
                        # logger.add_metric("q_t", self._impl._q_t)
                        # logger.add_metric("v_t", self._impl._v_t)
                        logger.add_metric("q_loss", self._impl._q_loss.detach().cpu().numpy())
                        logger.add_metric("v_loss", self._impl._v_loss.detach().cpu().numpy())
                        epoch_loss["q_loss"].append(self._impl._q_loss.detach().cpu().numpy())
                        epoch_loss["v_loss"].append(self._impl._v_loss.detach().cpu().numpy())
                        logger.add_metric("replay_q_loss", self._impl._replay_q_loss.detach().cpu().numpy())
                        logger.add_metric("replay_v_loss", self._impl._replay_v_loss.detach().cpu().numpy())
                        epoch_loss["replay_q_loss"].append(self._impl._replay_q_loss.detach().cpu().numpy())
                        epoch_loss["replay_v_loss"].append(self._impl._replay_v_loss.detach().cpu().numpy())
                        # logger.add_metric("replay_weight", self._impl._replay_weight)
                        # logger.add_metric("replay_log_probs", self._impl._replay_log_probs)
                        # logger.add_metric("replay_q_t", self._impl._replay_q_t)
                        # logger.add_metric("replay_v_t", self._impl._replay_v_t)
                        # logger.add_metric("replay_q_loss", self._impl._replay_q_loss)
                        # logger.add_metric("replay_v_loss", self._impl._replay_v_loss)
                    except AttributeError:
                        pass

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # save loss to loss history dict
            if epoch_num is None:
                self._loss_history["epoch"].append(epoch)
            else:
                self._loss_history["epoch"].append(epoch_num)
            self._loss_history["step"].append(total_step)
            for name, vals in epoch_loss.items():
                if vals:
                    self._loss_history[name].append(np.mean(vals))

            if scorers_list and eval_episodes_list:
                for scorer_num, (scorers, eval_episodes) in enumerate(zip(scorers_list, eval_episodes_list)):
                    rename_scorers = dict()
                    for name, scorer in scorers.items():
                        rename_scorers[str(scorer_num) + '_' + name] = scorer
                    #print("test predict: {self._impl.predict_best_action()}")
                    self._evaluate(eval_episodes, rename_scorers, logger)
            else:
                print(f"scorers_list: {scorers_list}")
                print(f"eval_episodes_list: {eval_episodes_list}")
                assert False

            # save metrics
            logger.commit(epoch, total_step)

            if save_epochs is not None and save_path is not None and epoch in save_epochs:
                torch.save({'buffer': None, 'algo': self}, save_path.replace(str(n_epochs * n_steps_per_epoch), str(epoch * n_steps_per_epoch), 1))

            # save model parameters
        logger.close()

    def fit_online(
        self,
        env: gym.envs,
        eval_env: gym.envs,
        buffer: Optional[Buffer],
        old_buffer: Optional[Buffer] = None,
        buffer_mix_ratio: float = 0.5,
        continual_type: str = "er",
        buffer_mix_type: str = "all",
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        start_epoch: int = 0,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_step: int = 100000,
        eval_epsilon: float = 0.0,
        save_metrics: bool = True,
        save_steps: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = False,
        tensorboard_dir: Optional[str] = None,
        scorers_list: Optional[
                List[Dict[str, Callable[[Any, List[Episode]], float]]]
            ] = None,
        eval_episodes_list: Optional[Dict[int, List[Episode]]] = None,
        callback: Optional[Callable[[LearnableBase, int, int], None]] = None,
        test: bool = False,
        # train_dynamics = False,
    ):
        """Trains with the given dataset.
        .. code-block:: python
            algo.fit(episodes, n_steps=1000000)
        Args:
            dataset: list of episodes to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        Returns:
            list of result tuples (epoch, metrics) per epoch.
        """
        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__ + "_online"

        logger = D3RLPyLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            with_timestamp=with_timestamp,
        )

        if buffer is None:
            buffer = ReplayBuffer(n_steps, env)

        if self._impl is None:
            LOG.debug("Building models...")
            observation_shape = env.observation_space.shape
            self.create_impl(
                self._process_observation_shape(observation_shape),
                dataset.get_action_size(),
            )
            self._impl._impl_id = 0
            LOG.debug("Models have been built.")
        else:
            # self._impl.rebuild_critic()
            self._impl._impl_id = 0
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        self.save_params(logger)

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        ## setup evaluation scorer
        #eval_scorer: Optional[Callable[..., float]]
        #if eval_env:
        #    eval_scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
        #else:
        #    eval_scorer = None
        # start training loop
        observation, _ = env.reset()
        if random_step > 0:
            exploit_observation, _ = eval_env.reset()
        rollout_return = 0.0

        for total_step in xrange(1, n_steps + 1):
            if total_step > 2000 and test:
                break
            with logger.measure_time("step"):
                #observation = observation.astype("f4")
                #fed_observation = observation

                # sample exploration action
                with logger.measure_time("inference"):
                    if total_step < random_step and not test:
                        action = env.action_space.sample()
                    else:
                        action = self.sample_action(observation[np.newaxis, :])
                        action = action[0]
                    #exploit_action = self.predict(observation[np.newaxis, :])
                    #exploit_action = exploit_action[0]

                # step environment
                episode_length = 0
                with logger.measure_time("environment_step"):
                    #exploit_next_observation, exploit_reward, exploit_terminal, exploit_truncated, exploit_info = eval_env.step(exploit_action)
                    next_observation, reward, terminal, truncated, info = env.step(action)
                    rollout_return += reward
                    episode_length += 1

                # special case for TimeLimit wrapper
                if truncated:
                    clip_episode = True
                    terminal = False
                    episode_length = 0
                else:
                    episode_length += 1
                    if episode_length == 1000 - 1:
                        terminal = True
                        episode_length = 0
                    clip_episode = terminal

                # store observation
                buffer.append(
                    observation=observation,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    clip_episode=clip_episode,
                )

                # reset if terminated
                if clip_episode:
                    observation, _ = env.reset()
                    logger.add_metric("rollout_return", rollout_return)
                    rollout_return = 0.0
                    # for image observation
                else:
                    observation = next_observation

                # psuedo epoch count
                epoch = total_step // n_steps_per_epoch + start_epoch

                if total_step > update_start_step and len(buffer) > self._batch_size:
                    if total_step % update_interval == 0:
                        # sample mini-batch
                        with logger.measure_time("sample_batch"):
                            if old_buffer is not None:
                                new_batch = buffer.sample(
                                    batch_size=self._batch_size,#round((1 - buffer_mix_ratio) * self._batch_size),
                                    n_frames=self._n_frames,
                                    n_steps=self._n_steps,
                                    gamma=self._gamma,
                                )
                                part_new_batch = OldTransitionMiniBatch(new_batch.transitions[:round((1 - buffer_mix_ratio) * self._batch_size)])
                                old_batch = old_buffer.sample(
                                    batch_size=round(buffer_mix_ratio * self._batch_size),
                                    n_frames=self._n_frames,
                                    n_steps=self._n_steps,
                                    gamma=self._gamma,
                                )
                                mix_batch = OldTransitionMiniBatch(
                                        part_new_batch.transitions + old_batch.transitions
                                        )
                                if buffer_mix_type == 'all':
                                    value_batch = policy_batch = mix_batch
                                elif buffer_mix_type == 'policy':
                                    policy_batch = mix_batch
                                    value_batch = new_batch
                                elif buffer_mix_type == 'value':
                                    policy_batch = new_batch
                                    value_batch = mix_batch
                                else:
                                    raise NotImplementedError
                            else:
                                policy_batch = buffer.sample(
                                    batch_size=self._batch_size,
                                    n_frames=self._n_frames,
                                    n_steps=self._n_steps,
                                    gamma=self._gamma,
                                )
                                value_batch = policy_batch

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self.update(policy_batch, value_batch, online=True)

                        # record metrics
                        for name, val in loss.items():
                            logger.add_metric(name, val)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if epoch > start_epoch and total_step % n_steps_per_epoch == 0:
                if scorers_list and eval_episodes_list:
                    for scorer_num, (scorers, eval_episodes) in enumerate(zip(scorers_list, eval_episodes_list)):
                        rename_scorers = dict()
                        for names, scorer in scorers.items():
                            names = names.split('-')
                            new_name = ""
                            for name_id, name in enumerate(names):
                                new_name += str(scorer_num) + '_' + name
                                if name_id != len(names) - 1:
                                    new_name += "-"
                            rename_scorers[new_name] = scorer
                        #print("test predict: {self._impl.predict_best_action()}")
                        self._evaluate(eval_episodes, rename_scorers, logger)

                # save metrics
                logger.commit(epoch, total_step)

            if save_steps is not None and save_path is not None and total_step in save_steps:
                buffer.clip_episode()
                torch.save({'buffer': buffer.to_mdp_dataset(), 'algo': self}, save_path)

        # clip the last episode
        buffer.clip_episode()

        # close logger
        logger.close()

    def before_learn(self, iterator, continual_type, buffer_mix_type, test):
        if continual_type in ['packnet'] and buffer_mix_type in ['all', 'value']:
            self._impl.critic_packnet_pre_train_process(iterator, self._batch_size, self._n_frames, self._n_steps, self._gamma, test=test)
        if continual_type in ['packnet'] and buffer_mix_type in ['all', 'policy']:
            self._impl.actor_packnet_pre_train_process(iterator, self._batch_size, self._n_frames, self._n_steps, self._gamma, test=test)

    def after_learn(self, iterator, continual_type, buffer_mix_type, test):
        if continual_type in ['rwalk_same', 'rwalk_all', 'ewc_same', 'ewc_all'] and buffer_mix_type in ['all', 'value']:
            self._impl.critic_ewc_rwalk_post_train_process(iterator, self._batch_size, self._n_frames, self._n_steps, self._gamma, test=test)
        if continual_type in ['rwalk_same', 'rwalk_all', 'ewc_same', 'ewc_all'] and buffer_mix_type in ['all', 'policy']:
            self._impl.actor_ewc_rwalk_post_train_process(iterator, self._batch_size, self._n_frames, self._n_steps, self._gamma, test=test)

    def copy_from_past(self, arg0: str, impl: STImpl, copy_optim: bool):
        assert self._impl is not None
        if arg0 in ['td3', 'td3_plus_bc']:
            self._impl.copy_from_td3(impl, copy_optim)
        elif arg0 in ['iql', 'iqln']:
            self._impl.copy_from_iql(impl, copy_optim)
        elif arg0 == 'sac':
            self._impl.copy_from_sac(impl, copy_optim)
        elif arg0 in ['cql', 'cal']:
            self._impl.copy_from_cql(impl, copy_optim)
        else:
            raise NotImplementedError

    def _evaluate(
        self,
        episodes: List[Episode],
        scorers: Dict[str, Callable[[Any, List[Episode]], Union[float, List[float]]]],
        logger: D3RLPyLogger,
    ) -> None:
        for names, scorer in scorers.items():
            names = names.split('-')
            # evaluation with test data
            test_scores = scorer(self, episodes)
            if not isinstance(test_scores, list):
                test_scores = [test_scores]
            assert len(names) == len(test_scores)
            for name, test_score in zip(names, test_scores):

                # logging metrics
                logger.add_metric(name, test_score)

                # store metric locally
                if test_score is not None:
                    self._eval_results[name].append(test_score)
