import time
import os
import copy

import numpy as np
import tensorflow as tf
import multiprocessing
import Queue
import datetime

from network import Network

""""
class EpisodeRunner:
    def __init__(self, config, gym_env, actor, is_in_collab=False):
        self.config = config
        self.gym_env = gym_env
        self.actor = actor

        self.is_in_collab = is_in_collab
        if self.is_in_collab:
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(400, 300))
            display.start()

    def _get_sampled_action(self, action):
        totally_random = np.random.binomial(1, self.config['model']['random_action_probability'], 1)[0]
        if totally_random:
            # take a completely random action
            result = np.random.uniform(-1.0, 1.0, np.shape(action))
        else:
            # modify existing step
            result = action + np.random.normal(0.0, self.config['model']['random_noise_std'], np.shape(action))
        action_space = self.gym_env.action_space
        # clip the action
        result = np.maximum(result, action_space.low)
        result = np.minimum(result, action_space.high)
        return result

    def _render(self, render):
        if render:
            if self.is_in_collab:
                import matplotlib.pyplot as plt
                from IPython import display as ipythondisplay
                screen = self.gym_env.render(mode='rgb_array')

                plt.imshow(screen)
                ipythondisplay.clear_output(wait=True)
                ipythondisplay.display(plt.gcf())
            else:
                self.gym_env.render()
                time.sleep(0.01)
    
    def run_episode(self, sess, is_train, render=False):
        # the trajectory data structures to return
        states = [self.gym_env.reset()]
        self._render(render)
        actions = []
        rewards = []
        done = False
        # set the start state
        start_rollout_time = datetime.datetime.now()
        max_steps = self.config['general']['max_steps']
        for j in range(max_steps):
            action_mean = self.actor.predict_action([states[-1]], sess, use_online_network=True)[0]
            sampled_action = self._get_sampled_action(action_mean) if is_train else action_mean
            next_state, reward, done, _ = self.gym_env.step(sampled_action)
            next_state = np.squeeze(next_state)
            # set a new current state
            states.append(next_state)
            actions.append(sampled_action)
            rewards.append(reward)
            # visualize if needed
            self._render(render)
            # break if needed
            if done:
                break
        # return the trajectory along with query info
        assert len(states) == len(actions) + 1
        assert len(states) == len(rewards) + 1
        end_episode_time = datetime.datetime.now()
        rollout_time = end_episode_time-start_rollout_time
        return states, actions, rewards, done, rollout_time
    """
"""
class ActorProcess(multiprocessing.Process):
    def __init__(self, config, generate_episode_queue, result_queue, actor_specific_queue, env_generator):
        multiprocessing.Process.__init__(self)
        self.generate_episode_queue = generate_episode_queue
        self.result_queue = result_queue
        self.actor_specific_queue = actor_specific_queue
        self.config = config
        self.env_generator = env_generator
        # members to set at runtime
        self.env = None
        self.actor = None
        self.episode_runner = None

    def _run_main_loop(self, sess):
        while True:
            try:
                # wait 1 second for a trajectory request
                is_train = self.generate_episode_queue.get(block=True, timeout=1)
                episode_result = self.episode_runner.run_episode(sess, is_train)
                self.result_queue.put(episode_result)
                self.generate_episode_queue.task_done()
            except Queue.Empty:
                pass
            try:
                next_actor_specific_task = self.actor_specific_queue.get(block=True, timeout=0.001)
                task_type = next_actor_specific_task[0]
                if task_type == 0:
                    # need to init the actor, called once.
                    assert self.actor is None
                    # on init, we only create a part of the graph (online actor model)
                    self.actor = Network(
                        self.config, self.env.get_state_space_dim(), self.env.get_action_space_dim(),
                        self.env.get_action_bounds(), True
                    )
                    sess.run(tf.global_variables_initializer())
                    # now initialize the episode runner
                    self.episode_runner = EpisodeRunner(self.config, self.env.get_env(), self.actor)
                    self.actor_specific_queue.task_done()
                elif task_type == 1:
                    # need to terminate
                    self.actor_specific_queue.task_done()
                    break
                elif task_type == 2:
                    # update the weights
                    new_weights = next_actor_specific_task[1]
                    self.actor.set_actor_online_weights(sess, new_weights)
                    self.actor_specific_queue.task_done()
            except Queue.Empty:
                pass


    def run(self):
        # write pid to file
        actor_id = os.getpid()
        # actor_file = os.path.join(os.getcwd(), 'actor_{}.sh'.format(actor_id))
        # with open(actor_file, 'w') as f:
        #     f.write('kill -9 {}'.format(actor_id))

        # init the env
        self.env = self.env_generator.get_env_wrapper()

        with tf.Session(
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.config['general']['actor_gpu_usage'])
                )
        ) as sess:
            self._run_main_loop(sess)

"""
class RolloutManager:
    def __init__(self, config, env_generator, actor_processes=None, fixed_queries=None):
        self.episode_generation_queue = multiprocessing.JoinableQueue()
        self.episode_results_queue = multiprocessing.Queue()
        if actor_processes is None:
            actor_processes = config['general']['actor_processes']
        self.fixed_queries = fixed_queries

        self.actor_specific_queues = [multiprocessing.JoinableQueue() for _ in range(actor_processes)]

        self.actors = [
            ActorProcess(
                copy.deepcopy(config), self.episode_generation_queue, self.episode_results_queue,
                self.actor_specific_queues[i], env_generator
            )
            for i in range(actor_processes)
        ]

        # start all the actor processes
        for a in self.actors:
            a.start()
        # for every actor process, post a message to initialize the actor network
        for actor_queue in self.actor_specific_queues:
            actor_queue.put((0, ))
            actor_queue.join()

    def generate_episodes(self, number_of_episodes, is_train):
        for i in range(number_of_episodes):
            # place in queue
            self.episode_generation_queue.put(is_train)

        self.episode_generation_queue.join()

        episodes = []
        while number_of_episodes:
            number_of_episodes -= 1
            episodes.append(self.episode_results_queue.get())

        return episodes

    def set_policy_weights(self, weights):
        message = (2, weights)
        self._post_private_message(message, self.actor_specific_queues)

    def end(self):
        message = (1, )
        self._post_private_message(message, self.actor_specific_queues)

    @staticmethod
    def _post_private_message(message, queues):
        for queue in queues:
            queue.put(message)
        for queue in queues:
            queue.join()
