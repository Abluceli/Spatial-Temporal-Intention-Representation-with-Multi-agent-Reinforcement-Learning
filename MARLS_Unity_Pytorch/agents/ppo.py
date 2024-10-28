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

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, image_shape, lidar_shape, hidden_dim=128):
        """
        :param image_shape:
        :param lidar_shape:
        :param hidden_dim:
        """
        super(ValueNetwork, self).__init__()
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''2.激光雷达数据编码, 使用MLP'''
        self.lidar_encode = nn.Sequential(
            nn.Linear(lidar_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''1.Q值计算'''
        self.Q_value = nn.Linear(2 * hidden_dim, 1)

    def forward(self, image, lidar):
        """
        :param image: (batch_size, 3, 84, 84)
        :param lidar: (batch_size, lidar_shape)
        :return:
        """
        image_encode = self.conv(image)  # (batch_size, hidden_dim)
        lidar_encode = self.lidar_encode(lidar)  # (batch_size, hidden_dim)
        x = torch.cat([image_encode, lidar_encode], dim=-1)  # (batch_size, 2*hidden_dim)
        x = self.Q_value(x)  # (batch_size, 1)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, image_shape, lidar_shape, action_shape, hidden_dim, log_std_min=-20, log_std_max=0.5,
                 continue_action=False):
        """
        :param image_shape:
        :param lidar_shape:
        :param action_shape:
        :param hidden_dim:
        :param log_std_min:
        :param log_std_max:
        :param continue_action:
        """
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.continue_action = continue_action
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''2.激光雷达数据编码, 使用MLP'''
        self.lidar_encode = nn.Sequential(
            nn.Linear(lidar_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''3.策略生成'''
        if continue_action:
            self.mean_linear = nn.Sequential(
                nn.Linear(2 * hidden_dim, action_shape),
                nn.Tanh()
            )
            self.log_std_linear = nn.Linear(2 * hidden_dim, action_shape)
        else:
            self.action_probs_linear = nn.Sequential(
                nn.Linear(2 * hidden_dim, action_shape),
                nn.Softmax(dim=-1)
            )

    def forward(self, image, lidar):
        """
        :param image: (batch_size, 3, 84, 84)
        :param lidar: (batch_size, lidar_shape)
        :return:
        """
        image_encode = self.conv(image)  # (batch_size, hidden_dim)
        lidar_encode = self.lidar_encode(lidar)  # (batch_size, hidden_dim)
        x = torch.cat([image_encode, lidar_encode], dim=-1)  # (batch_size, 2*hidden_dim)

        if self.continue_action:
            mean = self.mean_linear(x)
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            pi = torch.distributions.Normal(mean, torch.exp(log_std))
            return mean, log_std, pi
        else:
            action_probs = self.action_probs_linear(x)
            pi = torch.distributions.Categorical(action_probs)
            return action_probs, pi


class Agent():
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
                 resume=False):

        self.name = name
        self.obs_shape_list = obs_shape  # env.observation_space [12,12,12]
        self.act_type_list = act_type  # env.action_space [2,2,2]
        self.act_space_list = act_space  # env.action_space [2,2,2]
        self.agent_num = agent_num
        self.group_num = group_num
        self.agent_group_index = agent_group_index  # "agent_group_index":[0, 0, 0], #环境中每个agent对应的组编号，若3个智能体都在一队，则为[
        # 0,0,0]
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

        if self.share_parameters:
            self.actors = [PolicyNetwork(image_shape=(3, 84, 84),
                                         lidar_shape=self.group_obs_shape_list[group_index][1][0],
                                         action_shape=self.group_act_shape_list[group_index],
                                         hidden_dim=128,
                                         continue_action=self.group_act_type_list[group_index]).to(device)
                           for group_index in range(self.group_num)]
            self.critics = [ValueNetwork(image_shape=(3, 84, 84),
                                         lidar_shape=self.group_obs_shape_list[group_index][1][0],
                                         hidden_dim=128).to(device) for group_index in range(self.group_num)]

            self.actor_optimizers = [optim.Adam(self.actors[group_index].parameters(), lr=self.parameters["A_LR"])
                                     for group_index in range(self.group_num)]
            self.critic_optimizers = [optim.Adam(self.critics[group_index].parameters(), lr=self.parameters["C_LR"])
                                      for group_index in range(self.group_num)]
        else:
            self.actors = [PolicyNetwork(image_shape=(3, 84, 84),
                                         lidar_shape=self.obs_shape_list[agent_index][1][0],
                                         action_shape=self.act_space_list[agent_index],
                                         hidden_dim=128,
                                         continue_action=self.act_type_list[agent_index]).to(device)
                           for agent_index in range(self.agent_num)]
            self.critics = [ValueNetwork(image_shape=(3, 84, 84),
                                         lidar_shape=self.obs_shape_list[agent_index][1][0],
                                         hidden_dim=128).to(device) for agent_index in range(self.agent_num)]

            self.actor_optimizers = [optim.Adam(self.actors[agent_index].parameters(), lr=self.parameters["A_LR"])
                                     for agent_index in range(self.agent_num)]
            self.critic_optimizers = [optim.Adam(self.critics[agent_index].parameters(), lr=self.parameters["C_LR"])
                                      for agent_index in range(self.agent_num)]

        self.buffers = [{'image': [],
                         'lidar': [],
                         'action': [],
                         'reward': [],
                         'image_next': [],
                         'lidar_next': [],
                         'done': [],
                         'info': []
                         }
                        for _ in range(self.agent_num)]

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
                "summary_dir": '/PPO_Summary_' + str(current_time),
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

    def action(self, obs_n, evaluation=False):  # 所有智能体obs in,所有action out
        action_n = []
        for i, obs in enumerate(obs_n):
            image = torch.as_tensor([obs[0]], dtype=torch.float32, device=device)
            lidar = torch.as_tensor([obs[1]], dtype=torch.float32, device=device)
            if self.share_parameters:
                if self.group_act_type_list[self.agent_group_index[i]]:
                    mean, log_std, pi = self.actors[self.agent_group_index[i]](image=image, lidar=lidar)
                    if evaluation:
                        act = mean.cpu().detach().numpy()[0]
                    else:
                        act = pi.sample().cpu().detach().numpy()[0]
                    act = np.clip(act, -1., 1.)
                    action_n.append(act)
                else:
                    action_probs, pi = self.actors[self.agent_group_index[i]](image=image, lidar=lidar)
                    act = np.zeros(self.group_act_shape_list[self.agent_group_index[i]])
                    if evaluation:
                        act[torch.argmax(action_probs[0])] = 1
                    else:
                        act[pi.sample()[0]] = 1
                    action_n.append(act)
            else:
                if self.act_type_list[i]:
                    mean, log_std, pi = self.actors[i](image=image, lidar=lidar)
                    if evaluation:
                        act = mean.cpu().detach().numpy()[0]
                    else:
                        act = pi.sample().cpu().detach().numpy()[0]
                    act = np.clip(act, -1, 1)
                    action_n.append(act)
                else:
                    action_probs, pi = self.actors[i](image=image, lidar=lidar)
                    act = np.zeros(self.act_space_list[i])
                    if evaluation:
                        act[torch.argmax(action_probs[0])] = 1
                    else:
                        act[pi.sample()[0]] = 1
                    action_n.append(act)
        return action_n

    # 存储transition
    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n, info_n):
        for i in range(self.agent_num):
            self.buffers[i]['image'].append(obs_n[i][0])
            self.buffers[i]['lidar'].append(obs_n[i][1])
            self.buffers[i]['action'].append(act_n[i])
            self.buffers[i]['reward'].append([rew_n[i]])
            self.buffers[i]['image_next'].append(new_obs_n[i][0])
            self.buffers[i]['lidar_next'].append(new_obs_n[i][1])
            self.buffers[i]['done'].append([float(done_n[i]) - float(info_n[i])])
            self.buffers[i]['info'].append([float(info_n[i])])

    def save_model(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                torch.save(self.actors[group_index].state_dict(),
                           self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth")
                torch.save(self.critics[group_index].state_dict(),
                           self.model_path + "/ppo_critic_group_" + str(group_index) + ".pth")
        else:
            for agent_index in range(self.agent_num):
                torch.save(self.actors[agent_index].state_dict(),
                           self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth")
                torch.save(self.critics[agent_index].state_dict(),
                           self.model_path + "/ppo_critic_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            for group_index in range(self.group_num):
                if os.path.exists(self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth"):
                    try:
                        self.actors[group_index].load_state_dict(
                            torch.load(self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth"))
                        self.critics[group_index].load_state_dict(
                            torch.load(self.model_path + "/ppo_critic_group_" + str(group_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
        else:
            for agent_index in range(self.agent_num):
                if os.path.exists(self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth"):
                    try:
                        self.actors[agent_index].load_state_dict(
                            torch.load(self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth"))
                        self.critics[agent_index].load_state_dict(
                            torch.load(self.model_path + "/ppo_critic_agent_" + str(agent_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break

    def load_actor(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                self.actors[group_index].load_state_dict(
                    torch.load(self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth"))
        else:
            for agent_index in range(self.agent_num):
                self.actors[agent_index].load_state_dict(
                    torch.load(self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth"))

    def update(self, train_step):
        image_n = []
        lidar_n = []
        act_n = []
        reward_n = []
        image_next_n = []
        lidar_next_n = []
        done_n = []
        info_n = []
        old_pi_n = []

        for agent_index in range(self.agent_num):
            """1.获取数据"""
            image = torch.tensor(self.buffers[agent_index]['image'], dtype=torch.float32, device=device)
            lidar = torch.tensor(self.buffers[agent_index]['lidar'], dtype=torch.float32, device=device)
            act = torch.tensor(self.buffers[agent_index]['action'], dtype=torch.float32, device=device)
            reward = torch.tensor(self.buffers[agent_index]['reward'], dtype=torch.float32, device=device)
            image_next = torch.tensor(self.buffers[agent_index]['image_next'], dtype=torch.float32, device=device)
            lidar_next = torch.tensor(self.buffers[agent_index]['lidar_next'], dtype=torch.float32, device=device)
            done = torch.tensor(self.buffers[agent_index]['done'], dtype=torch.float32, device=device)
            info = torch.tensor(self.buffers[agent_index]['info'], dtype=torch.float32, device=device)
            image_n.append(image)
            lidar_n.append(lidar)
            act_n.append(act)
            reward_n.append(reward)
            image_next_n.append(image_next)
            lidar_next_n.append(lidar_next)
            done_n.append(done)
            info_n.append(info)
            """2.计算v(s_t),v(s_t+1)"""
            with torch.no_grad():
                if self.share_parameters:
                    if self.group_act_type_list[self.agent_group_index[agent_index]]:
                        mean, log_std, pi = self.actors[self.agent_group_index[agent_index]](image, lidar)
                    else:
                        action_probs, pi = self.actors[self.agent_group_index[agent_index]](image, lidar)
                else:
                    if self.act_type_list[agent_index]:
                        mean, log_std, pi = self.actors[agent_index](image, lidar)
                    else:
                        action_probs, pi = self.actors[agent_index](image, lidar)
            old_pi_n.append(pi)

        if self.share_parameters:
            summaries = {
                'LOSS/PPO_actor_loss': [0 for _ in range(self.group_num)],
                'LOSS/Entropy': [0 for _ in range(self.group_num)],
                'LOSS/PPO_critic_loss': [0 for _ in range(self.group_num)],
            }
        else:
            summaries = {
                'LOSS/PPO_actor_loss': [0 for _ in range(self.agent_num)],
                'LOSS/Entropy': [0 for _ in range(self.agent_num)],
                'LOSS/PPO_critic_loss': [0 for _ in range(self.agent_num)],
            }

        for _ in range(self.parameters['UPDATE_STEPS']):
            summary = self.actor_critic_train(
                memories=(image_n, lidar_n, act_n, reward_n, image_next_n, lidar_next_n, done_n, info_n, old_pi_n))
            for key in summary.keys():
                summaries[key] = [i + j for i, j in zip(summaries[key], summary[key])]

        # 每一轮训练结束，清空本轮所有的buffer
        for buffer in self.buffers:
            for key in buffer.keys():
                buffer[key].clear()

        for i in range(self.agent_num):
            if self.share_parameters:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key,
                                                       summaries[key][self.agent_group_index[i]] / self.parameters[
                                                           'UPDATE_STEPS'],
                                                       global_step=train_step)
            else:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key,
                                                       summaries[key][i] / self.parameters['UPDATE_STEPS'],
                                                       global_step=train_step)
            self.summary_writers[i].flush()

    def actor_critic_train(self, memories):
        image_n, lidar_n, act_n, reward_n, image_next_n, lidar_next_n, done_n, info_n, old_pi_n = memories

        actors_loss = [torch.tensor(0, dtype=torch.float32, device=device) for _ in range(self.agent_num)]
        mean_entropy = [torch.tensor(0, dtype=torch.float32, device=device) for _ in range(self.agent_num)]
        critics_loss = [torch.tensor(0, dtype=torch.float32, device=device) for _ in range(self.agent_num)]

        for agent_index in range(self.agent_num):
            value = self.critics[agent_index](image_n[agent_index], lidar_n[agent_index])
            value_next = self.critics[agent_index](image_next_n[agent_index], lidar_next_n[agent_index])

            """1.计算discount_reward"""
            discount_reward = []
            if done_n[agent_index][-1][0]:
                v_s_T = torch.tensor(np.asarray([0]), dtype=torch.float32, device=device)
            else:
                v_s_T = value_next[-1]
            i = reward_n[agent_index].shape[0] - 1
            while i >= 0:
                v_s_T = reward_n[agent_index][i] + self.parameters["gamma"] * v_s_T
                discount_reward.insert(0, v_s_T)
                i = i - 1
            discount_reward = torch.unsqueeze(torch.tensor(discount_reward, dtype=torch.float32, device=device), dim=-1)
            """2.计算advantage"""
            if self.parameters['use_gae_adv']:
                """3.计算delta"""
                delta = reward_n[agent_index] + self.parameters["gamma"] * value_next.detach() - value.detach()
                advantage = []
                adv = torch.tensor(np.asarray([0]), dtype=torch.float32, device=device)
                i = delta.shape[0] - 1
                while i >= 0:
                    adv = delta[i] + self.parameters["lambda"] * adv
                    advantage.insert(0, adv)
                    i = i - 1
                advantage = torch.unsqueeze(torch.tensor(advantage, dtype=torch.float32, device=device), dim=-1)
            else:
                advantage = discount_reward.detach() - value.detach()

            # actor_loss
            if self.act_type_list[agent_index]:
                mean, log_std, pi = self.actors[agent_index](image_n[agent_index], lidar_n[agent_index])
                ratio = torch.exp(pi.log_prob(act_n[agent_index]) - old_pi_n[agent_index].log_prob(act_n[agent_index]))
                surrogate = ratio * advantage
            else:
                action_probs, pi = self.actors[agent_index](image_n[agent_index], lidar_n[agent_index])
                ratio = torch.exp(pi.log_prob(torch.argmax(act_n[agent_index], dim=-1, keepdim=False)) - old_pi_n[
                    agent_index].log_prob(torch.argmax(act_n[agent_index], dim=-1, keepdim=False)))
                surrogate = ratio * torch.squeeze(advantage, dim=-1)
            entropy = pi.entropy()

            actor_loss = -torch.mean(torch.minimum(surrogate, torch.clamp(ratio,
                                                                          1. - self.parameters['epsilon'],
                                                                          1. + self.parameters['epsilon']) *
                                                   torch.squeeze(advantage, dim=-1)))
            entropy = self.parameters['entropy_dis'] * torch.mean(entropy)
            actor_loss = actor_loss - entropy

            self.actor_optimizers[agent_index].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_index].step()

            # critic_loss
            critic_loss = torch.mean(torch.square(discount_reward.detach() - value))

            self.critic_optimizers[agent_index].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_index].step()

            actors_loss[agent_index] = actor_loss
            mean_entropy[agent_index] = entropy
            critics_loss[agent_index] = critic_loss

        summaries = {
            'LOSS/PPO_actor_loss': actors_loss,
            'LOSS/Entropy': mean_entropy,
            'LOSS/PPO_critic_loss': critics_loss
        }
        return summaries
