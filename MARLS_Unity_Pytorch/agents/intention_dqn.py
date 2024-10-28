import itertools
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
from buffers.replay_buffer import ReplayBuffer_Intention_DQN

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MLP_net
class MLP_net(nn.Module):
    def __init__(self, image_shape, lidar_shape, intention_shape, action_shape, hidden_dim=64):
        '''
        :param image_shape:输入视觉信息的维度
        :param lidar_shape:输入激光雷达数据的维度
        :param intention_shape:输入意图检测信息的维度
        :param action_shape:动作维度
        :param hidden_dim:隐藏层维度
        '''
        super().__init__()

        '''1.原始图像数据编码, 使用图像卷积'''
        # (, 3, 256, 256) -> (, 32, 32, 32) -> (, 64, 8, 8)
        # (, 3, 84, 84) -> (, 32, 20, 20) -> (, 64, 5, 5)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(4, 4)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''2.激光雷达数据编码, 使用MLP'''
        self.lidar_encode = nn.Sequential(
            nn.Linear(lidar_shape, hidden_dim),
            nn.ReLU())  # batch_size, hidden_dim

        '''3.目标意图数据编码, 使用MLP, 意图数据输入维度是重点需要注意的'''
        self.intention_encode_1 = nn.Sequential(
            nn.Linear(intention_shape, hidden_dim),
            nn.ReLU())  # batch_size, object_num, hidden_dim
        self.intention_encode_2 = nn.Sequential(
            nn.Linear(intention_shape, hidden_dim),
            nn.ReLU())  # batch_size, object_num, hidden_dim

        '''4.Q值计算'''
        self.Q_value = nn.Linear(3*hidden_dim, action_shape)

    def forward(self, image_info, lidar_info, intention_info):
        '''
        :param image_info: batch_size, 3x256x256
        :param lidar_info: batch_size, lidar_length
        :param intention_info: batch_size, object_num, 5
        :return:
        '''
        image_encoding = self.conv(image_info)  # batch_size, hidden_dim

        intention_h = self.intention_encode_1(intention_info)  # batch_size, object_num, hidden_dim
        intention_e = self.intention_encode_2(intention_info)  # batch_size, object_num, hidden_dim
        intention_a = F.softmax(intention_e, dim=1)  # batch_size, object_num, hidden_dim
        intention_o = torch.sum(intention_a * intention_h, dim=1)  # batch_size, hidden_dim

        lidar_encoding = self.lidar_encode(lidar_info)  # batch_size, hidden_dim

        info_cat = torch.cat([image_encoding, intention_o, lidar_encoding], dim=-1)  # batch_size, 3*hidden_dim

        q_v = self.Q_value(info_cat)  # batch_size, action_shape
        return q_v

# INTENTION_DQNAgent Class
class Agent():
    def __init__(self,
                 name,
                 obs_shape,
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
        self.obs_shape_list = obs_shape
        self.act_space_list = act_space
        # self.state_shape = sum(obs_shape)
        self.agent_num = agent_num
        self.group_num = group_num
        self.agent_group_index = agent_group_index
        self.share_parameters = share_parameters
        self.parameters = parameters
        self.model_path = model_path
        self.log_path = log_path
        self.epsilon = parameters['epsilon']
        self.epsilon_decay = parameters['epsilon_decay']
        self.epsilon_min = parameters['epsilon_min']

        self.group_obs_shape_list = [0 for i in range(self.group_num)]
        self.group_act_shape_list = [0 for i in range(self.group_num)]
        for agent_index, group_index in enumerate(self.agent_group_index):
            if self.group_obs_shape_list[group_index] == 0:
                self.group_obs_shape_list[group_index] = self.obs_shape_list[agent_index]
            if self.group_act_shape_list[group_index] == 0:
                self.group_act_shape_list[group_index] = self.act_space_list[agent_index]

        if self.share_parameters:
            '''Unity中目标检测结果的维度为5,[x, y, h, w, intention_type]'''
            self.eval_nets = [
                MLP_net(image_shape=self.group_obs_shape_list[group_index][0], 
                        lidar_shape=self.group_obs_shape_list[group_index][1][0],
                        intention_shape=5,
                        action_shape=self.group_act_shape_list[group_index],
                        hidden_dim=64).to(device) for group_index in range(self.group_num)]
            self.targets_nets = [
                MLP_net(image_shape=self.group_obs_shape_list[group_index][0], 
                        lidar_shape=self.group_obs_shape_list[group_index][1][0],
                        intention_shape=5,
                        action_shape=self.group_act_shape_list[group_index],
                        hidden_dim=64).to(device) for group_index in range(self.group_num)]

            self.optimizers = [optim.Adam(self.eval_nets[group_index].parameters(), lr=parameters['lr']) for group_index
                               in range(self.group_num)]

        else:
            # 神经网络
            self.eval_nets = [
                MLP_net(image_shape=self.obs_shape_list[agent_index][0],
                        lidar_shape=self.obs_shape_list[agent_index][1][0],
                        intention_shape=5,
                        action_shape=self.act_space_list[agent_index],
                        hidden_dim=64).to(device) for agent_index in range(self.agent_num)]
            self.targets_nets = [
                MLP_net(image_shape=self.obs_shape_list[agent_index][0],
                        lidar_shape=self.obs_shape_list[agent_index][1][0],
                        intention_shape=5,
                        action_shape=self.act_space_list[agent_index],
                        hidden_dim=64).to(device) for agent_index in range(self.agent_num)]

            self.optimizers = [optim.Adam(self.eval_nets[agent_index].parameters(), lr=parameters['lr']) for agent_index
                               in range(self.agent_num)]

        # Create experience buffer
        self.replay_buffers = [ReplayBuffer_Intention_DQN(self.parameters["buffer_size"]) for agent_index in range(self.agent_num)]
        self.max_replay_buffer_len = parameters['max_replay_buffer_len']
        self.replay_sample_index = None

        self.update_target_weights(tau=1)

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
                "summary_dir": '/Intention_DQN_Summary_' + str(current_time),
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
        for eval_nets, targets_nets in zip(self.eval_nets, self.targets_nets):
            for eval_param, target_param in zip(eval_nets.parameters(), targets_nets.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)

    def action(self, obs_n, evaluation=False):
        '''
        :param obs_n: agent_num x [image, lidar, intention]
        :param evaluation:
        :return:
        '''
        s_n = []
        for i, obs in enumerate(obs_n):
            o_n = []
            #image
            o_n.append(torch.tensor([obs[0]], dtype=torch.float32, device=device))
            #lidar
            o_n.append(torch.tensor([obs[1]], dtype=torch.float32, device=device))
            #intention
            o_n.append(torch.tensor([np.reshape(np.asarray(obs[2]), [8, 5])], dtype=torch.float32, device=device))

            s_n.append(o_n)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        action_n = []
        for i, obs in enumerate(s_n):
            action = np.zeros(self.act_space_list[i])
            if self.share_parameters:
                action_value = self.eval_nets[self.agent_group_index[i]](obs[0], obs[1], obs[2]).detach()
            else:
                action_value = self.eval_nets[i](obs[0], obs[1], obs[2]).detach()
            if evaluation:
                action[torch.argmax(action_value[0])] = 1
                action_n.append(action)
            else:
                if np.random.random() < self.epsilon:
                    action[np.argmax(np.random.random(self.act_space_list[i]))] = 1
                    action_n.append(action)
                else:
                    action[torch.argmax(action_value[0])] = 1
                    action_n.append(action)
        return action_n

    # ------------------------memory(Replay Buffer)的更新---------------------------------------------
    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n):
        # Store transition in the replay buffer.
        for i in range(self.agent_num):
            self.replay_buffers[i].add(
                                        image=obs_n[i][0],
                                        intention=np.reshape(np.asarray(obs_n[i][2]), [8, 5]),
                                        state=obs_n[i][1],
                                        action=act_n[i],
                                        reward=[rew_n[i]],
                                        image_=new_obs_n[i][0],
                                        intention_=np.reshape(np.asarray(new_obs_n[i][2]), [8, 5]),
                                        state_=new_obs_n[i][1],
                                        done=[float(done_n[i])]
                                        )


    def save_model(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                torch.save(self.eval_nets[group_index].state_dict(),
                           self.model_path + "/Intention_DQN_group_" + str(group_index) + ".pth")
        else:
            for agent_index in range(self.agent_num):
                torch.save(self.eval_nets[agent_index].state_dict(),
                           self.model_path + "/Intention_DQN_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            for group_index in range(self.group_num):
                if os.path.exists(self.model_path + "/Intention_DQN_agent_" + str(group_index) + ".pth"):
                    try:
                        self.eval_nets[group_index].load_state_dict(
                            torch.load(self.model_path + "/Intention_DQN_agent_" + str(group_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
        else:
            for agent_index in range(self.agent_num):
                if os.path.exists(self.model_path + "/Intention_DQN_agent_" + str(agent_index) + ".pth"):
                    try:
                        self.eval_nets[agent_index].load_state_dict(
                            torch.load(self.model_path + "/Intention_DQN_agent_" + str(agent_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break

    def load_actor(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                self.eval_nets[group_index].load_state_dict(
                    torch.load(self.model_path + "/Intention_DQN_group_" + str(group_index) + ".pth"))
        else:
            for agent_index in range(self.agent_num):
                self.eval_nets[agent_index].load_state_dict(
                    torch.load(self.model_path + "/Intention_DQN_agent_" + str(agent_index) + ".pth"))

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
        intention_n = []
        state_n = []
        action_n = []
        reward_n = []
        image__n = []
        intention__n = []
        state__n = []
        done_n = []

        for i in range(self.agent_num):
            image, intention, state, action, reward, image_, intention_, state_, done = self.replay_buffers[i].sample_index(replay_sample_index)
            image_n.append(torch.tensor(image, dtype=torch.float32, device=device))
            intention_n.append(torch.tensor(intention, dtype=torch.float32, device=device))
            state_n.append(torch.tensor(state, dtype=torch.float32, device=device))
            action_n.append(torch.tensor(action, dtype=torch.float32, device=device))
            reward_n.append(torch.tensor(reward, dtype=torch.float32, device=device))
            image__n.append(torch.tensor(image_, dtype=torch.float32, device=device))
            intention__n.append(torch.tensor(intention_, dtype=torch.float32, device=device))
            state__n.append(torch.tensor(state_, dtype=torch.float32, device=device))
            done_n.append(torch.tensor(done, dtype=torch.float32, device=device))



        summaries = self.train((
            image_n,
            intention_n,
            state_n,
            action_n,
            reward_n,
            image__n,
            intention__n,
            state__n,
            done_n))

        if train_step % 100 == 0:  # only update every 100 steps
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
        # 更新网络参数
        # update the parameters更新参数
        # 1.从Replay Buffer(memory)中抽取batch
        # 2.计算loss
        # 3.更新参数theta
        # 4.更新参数theta^
        image_n, intention_n, state_n, action_n, reward_n, image__n, intention__n, state__n, done_n = memories

        if self.share_parameters:
            q_loss_n = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]

            for agent_index in range(self.agent_num):
                q_eval = self.eval_nets[self.agent_group_index[agent_index]](image_n[agent_index], state_n[agent_index], intention_n[agent_index])  # q_eval:(batch_size, action_num)
                # print('q_evql', q_eval.shape)
                act_index = torch.argmax(action_n[agent_index], dim=-1, keepdim=True)
                chosen_q_eval = torch.gather(q_eval, 1, act_index)  # (batch_size, 1)
                # chosen_q_eval = torch.sum(torch.multiply(q_eval, act_n[agent_index]), dim=-1, keepdim=True)# (batch_size, 1)

                # gather(1,action),dim=1表示按照横向
                # 将eval_net表中每一行的特定位置的值，按照给定的索引（batch_action），引用值给q_eval
                q_next = self.targets_nets[self.agent_group_index[agent_index]](image__n[agent_index], state__n[agent_index], intention__n[agent_index])  # (batch_size, action_num)
                # print('q_next', q_next.shape)
                max_q_next = torch.max(q_next, dim=-1, keepdim=True)[0]  # (batch_size, 1)

                q_target = reward_n[agent_index] + self.parameters['gamma'] * max_q_next * (1 - done_n[agent_index])  # targets:(bacth_size, 1)

                q_loss = nn.MSELoss()(chosen_q_eval, q_target.detach())
                q_loss_n[self.agent_group_index[agent_index]] += q_loss

            for group_index in range(self.group_num):
                self.optimizers[group_index].zero_grad()  # 优化器
                q_loss_n[group_index].backward()
                # nn.utils.clip_grad_norm_(self.all_parameters, self.parameters['grad_norm_clip'])
                self.optimizers[group_index].step()

        else:
            q_loss_n = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]

            for agent_index in range(self.agent_num):
                q_eval = self.eval_nets[agent_index](image_n[agent_index], state_n[agent_index], intention_n[agent_index])  # q_eval:(batch_size, action_num)
                # print('q_evql', q_eval.shape)
                act_index = torch.argmax(action_n[agent_index], dim=-1, keepdim=True)
                chosen_q_eval = torch.gather(q_eval, 1, act_index)  # (batch_size, 1)
                # chosen_q_eval = torch.sum(torch.multiply(q_eval, act_n[agent_index]), dim=-1, keepdim=True)# (batch_size, 1)

                # gather(1,action),dim=1表示按照横向
                # 将eval_net表中每一行的特定位置的值，按照给定的索引（batch_action），引用值给q_eval
                q_next = self.targets_nets[agent_index](image__n[agent_index], state__n[agent_index], intention__n[agent_index])  # (batch_size, action_num)
                # print('q_next', q_next.shape)
                max_q_next = torch.max(q_next, dim=-1, keepdim=True)[0]  # (batch_size, 1)

                q_target = reward_n[agent_index] + self.parameters['gamma'] * max_q_next  # targets:(bacth_size, 1)
                q_loss = nn.MSELoss()(chosen_q_eval, q_target.detach())

                self.optimizers[agent_index].zero_grad()  # 优化器
                q_loss.backward()
                # nn.utils.clip_grad_norm_(self.all_parameters, self.parameters['grad_norm_clip'])
                self.optimizers[agent_index].step()
                q_loss_n[agent_index] = q_loss

        summaries = dict([['LOSS/loss', q_loss_n],])
        return summaries


