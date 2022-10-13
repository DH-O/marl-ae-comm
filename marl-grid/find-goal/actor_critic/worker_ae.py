from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time, os, sys, datetime
from collections import deque

import numpy as np

import torch
import torch.multiprocessing as mp

from loss import policy_gradient_loss

from util import ops
from util.misc import check_done
from util.decorator import within_cuda_device


class Worker(mp.Process):
    """
    A3C worker. Each worker is responsible for collecting data from the
    environment and updating the master network by supplying the gradients.
    The worker re-synchronizes the weight at ever iteration.

    Args:
        master: master network instance.
        net: network with same architecture as the master network
        env: environment
        worker_id: worker id. used for tracking and debugging.
        gpu_id: the cuda gpu device id used to initialize variables. the
            `within_cuda_device` decorator uses this.
        t_max: maximum number of steps to take before applying gradient update.
            Default: `20`
        gamma: hyperparameter for the reward decay.
            Default: `0.99`
        tau: gae hyperparameter.
    """

    def __init__(self, master, net, env, worker_id, gpu_id=0, t_max=20,
                 gamma=0.99, tau=1.0, ae_loss_k=1.0):
        super().__init__()

        self.worker_id = worker_id
        self.net = net
        self.env = env
        self.master = master
        self.t_max = t_max
        self.gamma = gamma
        self.tau = tau
        self.gpu_id = gpu_id
        self.reward_log = deque(maxlen=5)  # track last 5 finished rewards
        self.pfmt = 'policy loss: {} value loss: {} ' + \
                    'entropy loss: {} ae loss: {} reward: {} time now: {}'
        self.agents = [f'agent_{i}' for i in range(self.env.num_agents)]
        self.num_acts = 1
        self.ae_loss_k = ae_loss_k

    @within_cuda_device
    def get_trajectory(self, hidden_state, state_var, done):
        """
        extracts a trajectory using the current policy.

        The first three return values (traj, val, tval) have `num_acts` length.

        Args:
            hidden_state: last hidden state observed
            state_var: last state observed
            done: boolean value to determine whether the env should be reset
        Returns:
            trajectory: (pi, a, v, r) trajectory [state is not tracked]
            values: reversed trajectory values .. used for GAE.
            target_value: the last time-step value
            done: updated indicator
        """
        # mask first (environment) actions after an agent is done
        env_mask_idx = [None for _ in range(len(self.agents))]  #에이전트 수만큼 None생성

        trajectory = [[] for _ in range(self.num_acts)] # 이거 왜 range가 하나밖에 안 나올까요?

        while not check_done(done) and len(trajectory[0]) < self.t_max:
            plogit, value, hidden_state, comm_out, comm_ae_loss = self.net(
                state_var, hidden_state, env_mask_idx=env_mask_idx)     # net의 결과로 현재 타임스텝에서의 가치함수 결과값과 정책 등을 아웃풋으로 받는다.
            action, _, _, all_actions = self.net.take_action(plogit, comm_out)  # net에 의해 나온 정책에 따라 액션을 취했다.
            state, reward, done, info = self.env.step(all_actions)  # 방금 구한 액션에 따라 env에서 한 스텝 진행한다.
            state_var = ops.to_state_var(state)

            # assert self.num_acts == 1:
            trajectory[0].append((plogit, action, value, reward, comm_ae_loss))

            # mask unavailable env actions after individual done
            for agent_id, a in enumerate(self.agents):
                if info[a]['done'] and env_mask_idx[agent_id] is None:
                    env_mask_idx[agent_id] = [0, 1, 2, 3]

        # end condition
        if check_done(done):
            target_value = [{k: 0 for k in self.agents} for _ in range(
                self.num_acts)]
        else:
            with torch.no_grad():
                target_value = self.net(state_var,
                                        hidden_state,
                                        env_mask_idx=env_mask_idx)[1]   # net의 결과 중 밸류값만 쏙 빼왔다.
                if self.num_acts == 1:
                    target_value = [target_value]

        #  compute Loss: accumulate rewards and compute gradient
        values = [{k: None for k in self.agents} for _ in range(
            self.num_acts)] # self.num_acts가 1이네요?

        # GAE
        for k in self.agents:
            for aid in range(self.num_acts):
                values[aid][k] = [x[k] for x in list(
                    zip(*trajectory[aid]))[2]]
                values[aid][k].append(ops.to_torch(
                    [target_value[aid][k]]))    #마지막에 텐서 하나 붙이는데 무슨 목적인지 궁금하긴 하다. -> 아 타겟 벨류긴 하다. 근데 왜 굳이?
                values[aid][k].reverse()    #이래서 values[0]의 길이가 20개가 아니라 21개이다. 진짜진짜 신기하다.

        return trajectory, values, target_value, done

    @within_cuda_device
    def run(self):
        self.master.init_tensorboard()
        done = True
        reward_log = 0.

        global_start = time.time()
        while not self.master.is_done():    #이거 끝나면 진짜 학습이 다 끝난 상황. worker is done 출력되는 상황.
            # synchronize network parameters
            weight_iter = self.master.copy_weights(self.net)
            self.net.zero_grad()    #그레디언트 초기화. 싹 다 0으로 만드는 것 같다.

            # reset environment if new episode
            if check_done(done):
                state = self.env.reset()
                state_var = ops.to_state_var(state) #dict to state variable. 사실 글로벌 스테이트를 에이전트별로 쪼갰다고 생각해도 된다.
                hidden_state = None

                if self.net.is_recurrent:
                    hidden_state = self.net.init_hidden()

                done = False

                self.reward_log.append(reward_log)
                reward_log = 0.

            # extract trajectory
            trajectory, values, target_value, done = \
                self.get_trajectory(hidden_state, state_var, done)  # 당연히 에이전트 3개에 대한 값들 다 뱉어낸다.

            all_pls = [[] for _ in range(self.num_acts)]
            all_vls = [[] for _ in range(self.num_acts)]
            all_els = [[] for _ in range(self.num_acts)]

            comm_ae_losses = []

            # compute loss for each action
            loss = 0
            for aid in range(self.num_acts):
                traj = trajectory[aid]
                val = values[aid]
                tar_val = target_value[aid]

                # compute loss - computed backward
                traj.reverse()  #이거 values가 시간 역순인거에 맞춰준거다. 

                for agent in self.agents:
                    gae = torch.zeros(1, 1).cuda()
                    t_value = tar_val[agent]

                    pls, vls, els = [], [], []
                    for i, (pi_logit, action, value, reward, comm_ae_loss
                            ) in enumerate(traj):                   # 20개의 timestep에서 
                        comm_ae_losses.append(comm_ae_loss.item())  #텐서 변수에서 값만 가져오기!

                        # Agent A3C Loss
                        t_value = reward[agent] + self.gamma * t_value
                        advantage = t_value - value[agent]

                        # Generalized advantage estimation (GAE)
                        delta_t = reward[agent] + \
                                  self.gamma * val[agent][i].data - \
                                  val[agent][i + 1].data    # .data는 텐서의 데이터만 가져오는 것 같다. 그냥 값을 복사해온 것이라고 생각한다. 여기서 그냥 웃긴게 reverse를 했으니까 i와 i+1을 쓴거다. 원래는 i+1과 -여야 한다.
                        gae = gae * self.gamma * self.tau + delta_t #sigma GAE 그 식을 이렇게 표현한 것이다. 점화식 형태로.

                        tl, (pl, vl, el) = policy_gradient_loss(
                            pi_logit[agent], action[agent], advantage, gae=gae)

                        pls.append(ops.to_numpy(pl))    #policy loss
                        vls.append(ops.to_numpy(vl))    #value loss
                        els.append(ops.to_numpy(el))    #entropy

                        reward_log += reward[agent]
                        loss += (tl + comm_ae_loss * self.ae_loss_k)

                    all_pls[aid].append(np.mean(pls))
                    all_vls[aid].append(np.mean(vls))
                    all_els[aid].append(np.mean(els))

            # accumulate gradient locally
            loss.backward()

            # log training info to tensorboard
            if self.worker_id == 0:
                log_dict = {}
                for act_id, act in enumerate(['env', 'comm'][:self.num_acts]):
                    for agent_id, agent in enumerate(self.agents):
                        log_dict[f'{act}_policy_loss/{agent}'] = all_pls[
                            act_id][agent_id]
                        log_dict[f'{act}_value_loss/{agent}'] = all_vls[act_id][
                            agent_id]
                        log_dict[f'{act}_entropy/{agent}'] = all_els[act_id][
                            agent_id]
                    log_dict[f'policy_loss/{act}'] = np.mean(all_pls[act_id])
                    log_dict[f'value_loss/{act}'] = np.mean(all_vls[act_id])
                    log_dict[f'entropy/{act}'] = np.mean(all_els[act_id])
                log_dict['ae_loss'] = np.mean(comm_ae_losses)

                for k, v in log_dict.items():
                    self.master.writer.add_scalar(k, v, weight_iter)

            now = str(datetime.datetime.now()).split(".")
            # all_pls, all_vls, all_els shape == (num_acts, num_agents)
            progress_str = self.pfmt.format(
                np.around(np.mean(all_pls, axis=-1), decimals=5),
                np.around(np.mean(all_vls, axis=-1), decimals=5),
                np.around(np.mean(all_els, axis=-1), decimals=5),
                np.around(np.mean(comm_ae_losses), decimals=5),
                np.around(np.mean(self.reward_log), decimals=2),
                now[0]
            )

            self.master.apply_gradients(self.net)
            self.master.increment(progress_str)

        print(f'worker {self.worker_id} is done.')
        global_end = time.time()
        global_consumed_time_ = global_end - global_start
        global_consumed_time = str(datetime.timedelta(seconds=global_consumed_time_)).split(".")
        print(f'The total consumed time: {global_consumed_time[0]}')
        return
