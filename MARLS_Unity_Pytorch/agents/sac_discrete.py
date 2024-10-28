import datetime
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from buffers.replay_buffer import ReplayBuffer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftQNetwork(nn.Module):
    def __init__(self, image_shape, lidar_shape, action_shape, hidden_dim=128):
        super(SoftQNetwork, self).__init__()

        '''1.原始图像数据编码, 使用图像卷积'''
        # (, 3, 84, 84) -> (, 32, 83, 83) -> (, 32, 41, 41) -> (, 64, 40, 40) -> (, 64, 19, 19)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 19 * 19, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''2.激光雷达数据编码, 使用MLP'''
        self.lidar_encode = nn.Sequential(
            nn.Linear(lidar_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''3.Q值计算'''
        self.Q_value = nn.Linear(2 * hidden_dim, action_shape)

    def forward(self, image, lidar):
        """
        计算Q(s,a)
        :param image:
        :param lidar:
        :return:
        """
        image_encode = self.conv(image)  # (batch_size, hidden_dim)
        lidar_encode = self.lidar_encode(lidar)  # (batch_size, hidden_dim)
        x = torch.cat([image_encode, lidar_encode], dim=-1)  # (batch_size, 2*hidden_dim+action)
        x = self.Q_value(x)  # (batch_size, action_shape)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, image_shape, lidar_shape, action_shape, hidden_dim):
        super(PolicyNetwork, self).__init__()

        self.action_shape = action_shape
        '''1.原始图像数据编码, 使用图像卷积'''
        # (, 3, 84, 84) -> (, 32, 83, 83) -> (, 32, 41, 41) -> (, 64, 40, 40) -> (, 64, 19, 19)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 19 * 19, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''2.激光雷达数据编码, 使用MLP'''
        self.lidar_encode = nn.Sequential(
            nn.Linear(lidar_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''3.策略生成'''
        self.action_probs_linear = nn.Sequential(
            nn.Linear(2 * hidden_dim, action_shape),
            nn.Softmax(dim=-1))

    def forward(self, image, lidar):
        image_encode = self.conv(image)  # (batch_size, hidden_dim)
        lidar_encode = self.lidar_encode(lidar)  # (batch_size, hidden_dim)
        x = torch.cat([image_encode, lidar_encode], dim=-1)  # (batch_size, 2*hidden_dim)
        probs = self.action_probs_linear(x)
        return probs

    def evaluate(self, image, lidar, epsilon=1e-8):
        probs = self.forward(image, lidar)
        log_probs = torch.log(probs)
        # Avoid numerical instability. Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)
        return log_probs

    def get_action(self, image, lidar, deterministic=False):
        probs = self.forward(image, lidar)
        pi = torch.distributions.Categorical(probs)
        act = np.zeros(self.action_shape)
        if deterministic:
            act[torch.argmax(probs[0])] = 1
        else:
            act[pi.sample()[0]] = 1
        return act

class Agent():
    '''
    Soft Actor-Critic version 2
    using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
    add alpha loss compared with version 1
    paper: https://arxiv.org/pdf/1812.05905.pdf
    '''

    def __init__(self,
                 name,
                 obs_shape,
                 act_type,
                 act_space,
                 agent_num,
                 group_num,
                 agent_group_index,
                 share_parameters,
                 parameters,
                 model_path,
                 log_path,
                 create_summary_writer=False,
                 max_episode_num=50000,
                 resume=False):

        self.name = name
        self.obs_shape_list = obs_shape
        self.act_type_list = act_type
        self.act_space_list = act_space
        self.agent_num = agent_num
        self.group_num = group_num
        self.agent_group_index = agent_group_index
        self.share_parameters = share_parameters
        self.parameters = parameters
        self.model_path = model_path
        self.log_path = log_path

        self.group_obs_shape_list = [0 for i in range(self.group_num)]
        self.group_act_type_list = [False for i in range(self.group_num)]
        self.group_act_shape_list = [0 for i in range(self.group_num)]
        for agent_index, group_index in enumerate(self.agent_group_index):
            if self.group_obs_shape_list[group_index] == 0:
                self.group_obs_shape_list[group_index] = self.obs_shape_list[agent_index]
            if not self.group_act_type_list[group_index]:
                self.group_act_type_list[group_index] = self.act_type_list[agent_index]
            if self.group_act_shape_list[group_index] == 0:
                self.group_act_shape_list[group_index] = self.act_space_list[agent_index]


        self.soft_q_net1s = [SoftQNetwork(image_shape=(3, 84, 84),
                                          lidar_shape=self.obs_shape_list[agent_index][1][0],
                                          action_shape=self.act_space_list[agent_index],
                                          hidden_dim=64).to(device) for
                             agent_index in range(self.agent_num)]
        self.soft_q_net2s = [SoftQNetwork(image_shape=(3, 84, 84),
                                          lidar_shape=self.obs_shape_list[agent_index][1][0],
                                          action_shape=self.act_space_list[agent_index],
                                          hidden_dim=64).to(device) for
                             agent_index in range(self.agent_num)]
        self.target_soft_q_net1s = [SoftQNetwork(image_shape=(3, 84, 84),
                                                 lidar_shape=self.obs_shape_list[agent_index][1][0],
                                                 action_shape=self.act_space_list[agent_index],
                                                 hidden_dim=64,).to(device) for
                                    agent_index in range(self.agent_num)]
        self.target_soft_q_net2s = [SoftQNetwork(image_shape=(3, 84, 84),
                                                 lidar_shape=self.obs_shape_list[agent_index][1][0],
                                                 action_shape=self.act_space_list[agent_index],
                                                 hidden_dim=64).to(device) for
                                    agent_index in range(self.agent_num)]
        self.policy_nets = [PolicyNetwork(image_shape=(3, 84, 84),
                                          lidar_shape=self.obs_shape_list[agent_index][1][0],
                                          action_shape=self.act_space_list[agent_index],
                                          hidden_dim=64).to(device) for
                            agent_index in range(self.agent_num)]

        self.log_alphas = [torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device) for agent_index in
                           range(self.agent_num)]
        if self.parameters['dynamic_alpha']:
            # entropy = -log(1/|A|) = log |A|
            self.target_entropys = [-2.0 for agent_index in range(self.agent_num)]
            self.alphas = [self.parameters["alpha"] for agent_index in range(self.agent_num)]
            self.alpha_optimizers = [optim.Adam([self.log_alphas[agent_index]], lr=self.parameters["alpha_lr"]) for agent_index in range(self.agent_num)]
        else:
            self.alphas = [self.parameters["alpha"] for agent_index in range(self.agent_num)]

        self.soft_q_optimizer1s = [
            optim.Adam(self.soft_q_net1s[agent_index].parameters(), lr=self.parameters["soft_q_lr"]) for agent_index
            in range(self.agent_num)]
        self.soft_q_optimizer2s = [
            optim.Adam(self.soft_q_net2s[agent_index].parameters(), lr=self.parameters["soft_q_lr"]) for agent_index
            in range(self.agent_num)]
        self.policy_optimizers = [
            optim.Adam(self.policy_nets[agent_index].parameters(), lr=self.parameters["policy_lr"]) for agent_index
            in range(self.agent_num)]

        self.update_target_weights(tau=1)

        # Create experience buffer
        self.buffers = [ReplayBuffer(self.parameters["buffer_size"]) for agent_index in range(self.agent_num)]
        self.max_replay_buffer_len = self.parameters['max_replay_buffer_len']

        # 为每一个agent构建tensorboard可视化训练过程
        if resume:
            with open(self.log_path + '/log_info.txt', 'r') as load_f:
                self.log_info_json = json.load(load_f)
                load_f.close()

            if create_summary_writer:
                self.summary_writers = []
                for i in range(self.agent_num):
                    train_log_dir = self.log_path + self.log_info_json["summary_dir"] + "agent_" + str(i)
                    self.summary_writers.append(SummaryWriter(train_log_dir))
            else:
                pass
            self.load_model()
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_info_json = {
                "summary_dir": '/SAC_Summary_' + str(current_time),
                "epoch": 0,
                "train_step": 0,
                "log_episode": 0
            }
            if create_summary_writer:
                self.summary_writers = []
                for i in range(self.agent_num):
                    train_log_dir = self.log_path + self.log_info_json["summary_dir"] + "agent_" + str(i)
                    self.summary_writers.append(SummaryWriter(train_log_dir))
            else:
                pass

    # update network parameters
    def update_target_weights(self, tau=1):
        # update target networks
        for soft_q_net1, target_soft_q_net1, soft_q_net2, target_soft_q_net2 in zip(self.soft_q_net1s,
                                                                                    self.target_soft_q_net1s,
                                                                                    self.soft_q_net2s,
                                                                                    self.target_soft_q_net2s):
            for eval_param, target_param in zip(soft_q_net1.parameters(), target_soft_q_net1.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)
            for eval_param, target_param in zip(soft_q_net2.parameters(), target_soft_q_net2.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)

    def action(self, obs_n, evaluation=False, global_step=0):
        action_n = []
        for i, obs in enumerate(obs_n):
            image = torch.as_tensor([obs[0]], dtype=torch.float32, device=device)
            lidar = torch.as_tensor([obs[1]], dtype=torch.float32, device=device)
            mu = self.policy_nets[i].get_action(image, lidar, deterministic=evaluation)
            action_n.append(mu)
        return action_n

    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n, info_n):
        # Store transition in the replay buffer.
        for i in range(self.agent_num):
            self.buffers[i].add(
                image=obs_n[i][0],
                state=obs_n[i][1],
                action=act_n[i],
                reward=[rew_n[i]],
                image_=new_obs_n[i][0],
                state_=new_obs_n[i][1],
                done=[float(done_n[i]) - float(info_n[i])]
            )

    # save_model("models/maddpg_actor_agent_", "models/maddpg_critic_agent_")
    def save_model(self):
        for agent_index in range(self.agent_num):
            torch.save(self.policy_nets[agent_index].state_dict(),
                       self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth")

            torch.save(self.policy_nets[agent_index].state_dict(),
                       self.model_path + "/sac_soft_q_1_agent_" + str(agent_index) + ".pth")

            torch.save(self.policy_nets[agent_index].state_dict(),
                       self.model_path + "/sac_soft_q_2_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        for agent_index in range(self.agent_num):
            if os.path.exists(self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth"):
                try:
                    self.policy_nets[agent_index].load_state_dict(
                        torch.load(self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth"))
                    self.soft_q_net1s[agent_index].load_state_dict(
                        torch.load(self.model_path + "/sac_soft_q_1_agent_" + str(agent_index) + ".pth"))
                    self.soft_q_net2s[agent_index].load_state_dict(
                        torch.load(self.model_path + "/sac_soft_q_2_agent_" + str(agent_index) + ".pth"))
                except RuntimeError as e:
                    print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
            else:
                print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                break

    def load_actor(self):
        for agent_index in range(self.agent_num):
            self.policy_nets[agent_index].load_state_dict(
                torch.load(self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth"))

    def can_update(self):
        can_up = []
        for i in range(self.agent_num):
            if len(self.buffers[i]) > self.max_replay_buffer_len:
                can_up.append(True)
            else:
                can_up.append(False)
        return all(can_up)

    def update(self, train_step):
        replay_sample_index = self.buffers[0].make_index(self.parameters['batch_size'])

        # collect replay sample from all agents
        image_n = []
        lidar_n = []
        action_n = []
        reward_n = []
        image_next_n = []
        lidar_next_n = []
        done_n = []

        for i in range(self.agent_num):
            image, state, action, reward, image_, state_, done = self.buffers[i].sample_index(replay_sample_index)
            image_n.append(torch.tensor(image, dtype=torch.float32, device=device))
            lidar_n.append(torch.tensor(state, dtype=torch.float32, device=device))
            action_n.append(torch.tensor(action, dtype=torch.float32, device=device))
            reward_n.append(torch.tensor(reward, dtype=torch.float32, device=device))
            image_next_n.append(torch.tensor(image_, dtype=torch.float32, device=device))
            lidar_next_n.append(torch.tensor(state_, dtype=torch.float32, device=device))
            done_n.append(torch.tensor(done, dtype=torch.float32, device=device))

        summaries = self.train((image_n, lidar_n, action_n, reward_n, image_next_n, lidar_next_n, done_n))

        # if train_step % 10 == 0:  # only update every 100 steps
        self.update_target_weights(tau=self.parameters["tau"])

        for i in range(self.agent_num):
            for key in summaries.keys():
                self.summary_writers[i].add_scalar(key, summaries[key][i], global_step=train_step)
            self.summary_writers[i].flush()

    def train(self, memories):
        image_n, lidar_n, action_n, reward_n, image_next_n, lidar_next_n, done_n = memories

        q1s_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
        q2s_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
        policys_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
        alphas_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]

        for agent_index in range(self.agent_num):
            predicted_q_value1 = self.soft_q_net1s[agent_index](image_n[agent_index], lidar_n[agent_index])
            predicted_q_value1 = torch.gather(predicted_q_value1, dim=-1, index=torch.argmax(action_n[agent_index], dim=-1, keepdim=True))
            predicted_q_value2 = self.soft_q_net2s[agent_index](image_n[agent_index], lidar_n[agent_index])
            predicted_q_value2 = torch.gather(predicted_q_value2, dim=-1, index=torch.argmax(action_n[agent_index], dim=-1, keepdim=True))
            log_prob = self.policy_nets[agent_index].evaluate(image_n[agent_index], lidar_n[agent_index])
            next_log_prob = self.policy_nets[agent_index].evaluate(image_next_n[agent_index], lidar_next_n[agent_index])
            # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

            self.alphas[agent_index] = self.log_alphas[agent_index].exp()

            # Training Q Function
            target_q_min = (next_log_prob.exp() * (torch.min(self.target_soft_q_net1s[agent_index](image_next_n[agent_index], lidar_next_n[agent_index]),
                                                             self.target_soft_q_net2s[agent_index](image_next_n[agent_index], lidar_next_n[agent_index]))
                                                   - self.alphas[agent_index] * next_log_prob)).sum(dim=-1).unsqueeze(-1)
            target_q_value = reward_n[agent_index] + (1 - done_n[agent_index]) * self.parameters["gamma"] * target_q_min  # if done==1, only reward
            q_value_loss1 = nn.MSELoss()(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
            q_value_loss2 = nn.MSELoss()(predicted_q_value2, target_q_value.detach())

            self.soft_q_optimizer1s[agent_index].zero_grad()
            q_value_loss1.mean().backward()
            self.soft_q_optimizer1s[agent_index].step()
            self.soft_q_optimizer2s[agent_index].zero_grad()
            q_value_loss2.mean().backward()
            self.soft_q_optimizer2s[agent_index].step()

            # Training Policy Function
            predicted_new_q_value = torch.min(self.soft_q_net1s[agent_index](image_n[agent_index], lidar_n[agent_index]),
                                              self.soft_q_net2s[agent_index](image_n[agent_index], lidar_n[agent_index]))
            policy_loss = (log_prob.exp() * (self.alphas[agent_index] * log_prob - predicted_new_q_value)).sum(dim=-1).mean()

            self.policy_optimizers[agent_index].zero_grad()
            policy_loss.backward()
            self.policy_optimizers[agent_index].step()


            if self.parameters['dynamic_alpha']:
                # Updating alpha wrt entropy
                # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
                alpha_loss = -(self.log_alphas[agent_index] * (log_prob + self.target_entropys[agent_index]).detach()).mean()
                # print('alpha loss: ',alpha_loss)
                self.alpha_optimizers[agent_index].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizers[agent_index].step()
            else:
                self.alphas[agent_index] = 1.
                alpha_loss = 0.0


            policys_loss[agent_index] = policy_loss
            q1s_loss[agent_index] = q_value_loss1
            q2s_loss[agent_index] = q_value_loss2
            alphas_loss[agent_index] = alpha_loss

        summaries = {
                'Loss/policy_loss': policys_loss,
                'Loss/q1_loss': q1s_loss,
                'Loss/q2_loss': q2s_loss,
                'Loss/alpha_loss': alphas_loss,
                'Loss/alpha': self.alphas
            }
        return summaries
