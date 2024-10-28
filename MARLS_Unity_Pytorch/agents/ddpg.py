import json
import os
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from buffers.replay_buffer import ReplayBuffer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor network
class Actor(nn.Module):
    def __init__(self, image_shape, lidar_shape, action_shape, hidden_dim):
        """
        :param image_shape:
        :param lidar_shape:
        :param action_shape:
        :param hidden_dim:
        :param log_std_min:
        :param log_std_max:
        :param continue_action:
        """
        super(Actor, self).__init__()
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
        self.action_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, action_shape),
            nn.Tanh()
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
        action = self.action_layer(x)
        return action


# Critic Model
class Critic(nn.Module):
    def __init__(self, image_shape, lidar_shape, action_shape, hidden_dim=128):
        super(Critic, self).__init__()

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

        '''3.Q值计算'''
        self.Q_value = nn.Sequential(
            nn.Linear(2 * hidden_dim + action_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, image, lidar, action):
        """
        计算Q(s,a)
        :param image:
        :param lidar:
        :param action:
        :return:
        """
        image_encode = self.conv(image)  # (batch_size, hidden_dim)
        lidar_encode = self.lidar_encode(lidar)  # (batch_size, hidden_dim)
        x = torch.cat([image_encode, lidar_encode, action], dim=-1)  # (batch_size, 2*hidden_dim+action)
        x = self.Q_value(x)  # (batch_size, 1)
        return x


# DDPGAgent Class
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
                 max_episode_num=10000,
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

        if self.share_parameters:
            self.actors = [Actor(image_shape=(3, 84, 84),
                                 lidar_shape=self.group_obs_shape_list[group_index][1][0],
                                 action_shape=self.group_act_shape_list[group_index],
                                 hidden_dim=64).to(device) for group_index in range(self.group_num)]
            self.critics = [Critic(image_shape=(3, 84, 84),
                                   lidar_shape=self.group_obs_shape_list[group_index][1][0],
                                   action_shape=self.group_act_shape_list[group_index],
                                   hidden_dim=64).to(device) for group_index in range(self.group_num)]

            self.actor_targets = [Actor(image_shape=(3, 84, 84),
                                        lidar_shape=self.group_obs_shape_list[group_index][1][0],
                                        action_shape=self.group_act_shape_list[group_index],
                                        hidden_dim=64).to(device) for group_index in range(self.group_num)]
            self.critic_targets = [Critic(image_shape=(3, 84, 84),
                                          lidar_shape=self.group_obs_shape_list[group_index][1][0],
                                          action_shape=self.group_act_shape_list[group_index],
                                          hidden_dim=64).to(device) for group_index in range(self.group_num)]

            self.actor_optimizers = [optim.Adam(self.actors[agent_index].parameters(), lr=self.parameters["lr_actor"])
                                     for agent_index in range(self.group_num)]
            self.critic_optimizers = [
                optim.Adam(self.critics[agent_index].parameters(), lr=self.parameters["lr_critic"])
                for agent_index in range(self.group_num)]

        else:
            self.actors = [Actor(image_shape=(3, 84, 84),
                                 lidar_shape=self.obs_shape_list[agent_index][1][0],
                                 action_shape=self.act_space_list[agent_index],
                                 hidden_dim=64).to(device) for agent_index in range(self.agent_num)]
            self.critics = [Critic(image_shape=(3, 84, 84),
                                   lidar_shape=self.obs_shape_list[agent_index][1][0],
                                   action_shape=self.act_space_list[agent_index],
                                   hidden_dim=64).to(device) for agent_index in range(self.agent_num)]

            self.actor_targets = [Actor(image_shape=(3, 84, 84),
                                        lidar_shape=self.obs_shape_list[agent_index][1][0],
                                        action_shape=self.act_space_list[agent_index],
                                        hidden_dim=64).to(device) for agent_index in range(self.agent_num)]
            self.critic_targets = [Critic(image_shape=(3, 84, 84),
                                          lidar_shape=self.obs_shape_list[agent_index][1][0],
                                          action_shape=self.act_space_list[agent_index],
                                          hidden_dim=64).to(device) for agent_index in range(self.agent_num)]

            self.actor_optimizers = [optim.Adam(self.actors[agent_index].parameters(), lr=self.parameters["lr_actor"])
                                     for agent_index in range(self.agent_num)]
            self.critic_optimizers = [
                optim.Adam(self.critics[agent_index].parameters(), lr=self.parameters["lr_critic"])
                for agent_index in range(self.agent_num)]

        self.action_noises = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_space_list[agent_index]),
                                                           sigma=self.parameters['sigma'])
                              for agent_index in range(self.agent_num)]

        self.update_target_weights(tau=1)

        # Create experience buffer
        self.replay_buffers = [ReplayBuffer(self.parameters["buffer_size"]) for agent_index in range(self.agent_num)]
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
                "summary_dir": '/DDPG_Summary_' + str(current_time),
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
        for actor, actor_target, critic, critic_target in zip(self.actors, self.actor_targets, self.critics,
                                                              self.critic_targets):
            for eval_param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)
            for eval_param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)

    def action(self, obs_n, evaluation=False, global_step=0):
        action_n = []
        for i, obs in enumerate(obs_n):
            image = torch.as_tensor([obs[0]], dtype=torch.float32, device=device)
            lidar = torch.as_tensor([obs[1]], dtype=torch.float32, device=device)
            if self.share_parameters:
                mu = self.actors[self.agent_group_index[i]](image, lidar).cpu().data.numpy()
            else:
                mu = self.actors[i](image, lidar).cpu().data.numpy()
            noise = np.asarray([self.action_noises[i]() for j in range(mu.shape[0])])
            # print(noise)
            pi = np.clip(mu + noise, -1, 1)
            a = mu if evaluation else pi
            action_n.append(np.array(a[0]))
        return action_n

    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n, info_n):
        # Store transition in the replay buffer.
        for i in range(self.agent_num):
            self.replay_buffers[i].add(
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
        if self.share_parameters:
            for group_index in range(self.group_num):
                torch.save(self.actors[group_index].state_dict(),
                           self.model_path + "/ddpg_actor_group_" + str(group_index) + ".pth")
                torch.save(self.critics[group_index].state_dict(),
                           self.model_path + "/ddpg_critic_group_" + str(group_index) + ".pth")
        else:
            for agent_index in range(self.agent_num):
                torch.save(self.actors[agent_index].state_dict(),
                           self.model_path + "/ddpg_actor_agent_" + str(agent_index) + ".pth")
                torch.save(self.critics[agent_index].state_dict(),
                           self.model_path + "/ddpg_critic_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            for group_index in range(self.group_num):
                if os.path.exists(
                        self.model_path + "/ddpg_actor_group_" + str(group_index) + ".pth") and os.path.exists(
                    self.model_path + "/ddpg_critic_group_" + str(group_index) + ".pth"):
                    try:
                        self.actors[group_index].load_state_dict(
                            torch.load(self.model_path + "/ddpg_actor_group_" + str(group_index) + ".pth"))
                        self.critics[group_index].load_state_dict(
                            torch.load(self.model_path + "/ddpg_critic_group_" + str(group_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
        else:
            for agent_index in range(self.agent_num):
                if os.path.exists(
                        self.model_path + "/ddpg_actor_agent_" + str(agent_index) + ".pth") and os.path.exists(
                    self.model_path + "/ddpg_critic_agent_" + str(agent_index) + ".pth"):
                    try:
                        self.actors[agent_index].load_state_dict(
                            torch.load(self.model_path + "/ddpg_actor_agent_" + str(agent_index) + ".pth"))
                        self.critics[agent_index].load_state_dict(
                            torch.load(self.model_path + "/ddpg_critic_agent_" + str(agent_index) + ".pth"))
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
                    torch.load(self.model_path + "/ddpg_actor_group_" + str(group_index) + ".pth"))
        else:
            for agent_index in range(self.agent_num):
                self.actors[agent_index].load_state_dict(
                    torch.load(self.model_path + "/ddpg_actor_agent_" + str(agent_index) + ".pth"))

    def can_update(self):
        can_up = []
        for i in range(self.agent_num):
            if len(self.replay_buffers[i]) > self.max_replay_buffer_len:
                can_up.append(True)
            else:
                can_up.append(False)
        return all(can_up)

    def update(self, train_step):
        replay_sample_index = self.replay_buffers[0].make_index(self.parameters['batch_size'])

        # collect replay sample from all agents
        image_n = []
        lidar_n = []
        action_n = []
        reward_n = []
        image_next_n = []
        lidar_next_n = []
        done_n = []
        action_next_n = []

        for i in range(self.agent_num):
            image, state, action, reward, image_, state_, done = self.replay_buffers[i].sample_index(
                replay_sample_index)
            image_n.append(torch.tensor(image, dtype=torch.float32, device=device))
            lidar_n.append(torch.tensor(state, dtype=torch.float32, device=device))
            action_n.append(torch.tensor(action, dtype=torch.float32, device=device))
            reward_n.append(torch.tensor(reward, dtype=torch.float32, device=device))
            image_next_n.append(torch.tensor(image_, dtype=torch.float32, device=device))
            lidar_next_n.append(torch.tensor(state_, dtype=torch.float32, device=device))
            done_n.append(torch.tensor(done, dtype=torch.float32, device=device))

        for i in range(self.agent_num):
            if self.share_parameters:
                target_mu = self.actor_targets[self.agent_group_index[i]](image_next_n[i], lidar_next_n[i])
            else:
                target_mu = self.actor_targets[i](image_next_n[i], lidar_next_n[i])
            # action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
            action_next_n.append(target_mu)

        summaries = self.train(
            (image_n, lidar_n, action_n, reward_n, image_next_n, lidar_next_n, done_n, action_next_n))

        # if train_step % 10 == 0:  # only update every 100 steps
        self.update_target_weights(tau=self.parameters["tau"])

        for i in range(self.agent_num):
            if self.share_parameters:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key, summaries[key][self.agent_group_index[i]],
                                                       global_step=train_step)
            else:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key, summaries[key][i], global_step=train_step)
            self.summary_writers[i].flush()

    def train(self, memories):
        image_n, lidar_n, action_n, reward_n, image_next_n, lidar_next_n, done_n, action_next_n = memories

        if self.share_parameters:
            q_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
            actor_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
            for agent_index in range(self.agent_num):
                # critic_loss
                q_target = self.critic_targets[self.agent_group_index[agent_index]](image_next_n[agent_index],
                                                                                    lidar_next_n[agent_index],
                                                                                    action_next_n[agent_index])
                q_target = reward_n[agent_index] + self.parameters['gamma'] * q_target * (1 - done_n[agent_index])
                q_eval = self.critics[self.agent_group_index[agent_index]](image_n[agent_index],
                                                                           lidar_n[agent_index],
                                                                           action_n[agent_index])
                q_loss[self.agent_group_index[agent_index]] += nn.MSELoss()(q_eval, q_target.detach())
            for group_index in range(self.group_num):
                # optimize critic
                self.critic_optimizers[group_index].zero_grad()
                q_loss[group_index].backward()
                self.critic_optimizers[group_index].step()

            for agent_index in range(self.agent_num):
                # actor_loss
                mu = self.actors[self.agent_group_index[agent_index]](image_n[agent_index], lidar_n[agent_index])
                actor_loss[self.agent_group_index[agent_index]] += -torch.mean(
                    self.critics[self.agent_group_index[agent_index]](image_n[agent_index], lidar_n[agent_index], mu))
            for group_index in range(self.group_num):
                # optimize actor
                self.actor_optimizers[group_index].zero_grad()
                actor_loss[group_index].backward()
                self.actor_optimizers[group_index].step()

        else:
            q_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            actor_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            for agent_index in range(self.agent_num):
                # critic_loss
                q_target = self.critic_targets[agent_index](image_next_n[agent_index], lidar_next_n[agent_index],
                                                            action_next_n[agent_index])
                q_target = reward_n[agent_index] + self.parameters['gamma'] * q_target * (1. - done_n[agent_index])
                q_eval = self.critics[agent_index](image_n[agent_index], lidar_n[agent_index], action_n[agent_index])
                q_l = nn.MSELoss()(q_eval, q_target.detach())

                self.critic_optimizers[agent_index].zero_grad()
                q_l.backward()
                self.critic_optimizers[agent_index].step()
                q_loss[agent_index] = q_l

                # actor_loss
                mu = self.actors[agent_index](image_n[agent_index], lidar_n[agent_index])
                a_l = -torch.mean(self.critics[agent_index](image_n[agent_index], lidar_n[agent_index], mu))

                self.actor_optimizers[agent_index].zero_grad()
                a_l.backward()
                self.actor_optimizers[agent_index].step()
                actor_loss[agent_index] = a_l

        summaries = {
            'Loss/actor_loss': actor_loss,
            'Loss/q_loss': q_loss
        }

        return summaries


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
