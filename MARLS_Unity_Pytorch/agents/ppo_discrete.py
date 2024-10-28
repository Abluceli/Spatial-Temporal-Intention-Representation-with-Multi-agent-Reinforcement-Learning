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
from agents.util import DecayedValue

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, image_shape, lidar_shape, intention_shape, hidden_dim=128, graphics=True):
        """
        :param image_shape:
        :param lidar_shape:
        :param hidden_dim:
        """
        super(ValueNetwork, self).__init__()
        self.graphics = graphics
        if self.graphics:
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
            nn.ReLU())  # batch_size, hidden_dim

        '''3.目标意图数据编码, 使用MLP, 意图数据输入维度是重点需要注意的'''
        self.intention_encode_1 = nn.Sequential(
            nn.Linear(intention_shape, hidden_dim),
            nn.ReLU())  # batch_size, object_num, hidden_dim
        self.intention_encode_2 = nn.Sequential(
            nn.Linear(intention_shape, hidden_dim),
            nn.ReLU())  # batch_size, object_num, hidden_dim

        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        self.Q_value = nn.Linear(3 * hidden_dim, 1)

    def forward(self, image, lidar, intention, hidden_state):
        """
        :param image: (batch_size, 3, 84, 84)
        :param lidar: (batch_size, lidar_shape)
        :return:
        """

        image_encode = self.conv(image)  # (batch_size, hidden_dim)
        lidar_encode = self.lidar_encode(lidar)  # (batch_size, hidden_dim)

        intention_h = self.intention_encode_1(intention)  # batch_size, object_num, hidden_dim
        intention_e = self.intention_encode_2(intention)  # batch_size, object_num, hidden_dim
        intention_a = F.softmax(intention_e, dim=1)  # batch_size, object_num, hidden_dim
        intention_o = torch.sum(intention_a * intention_h, dim=1)  # batch_size, hidden_dim
        h1 = torch.unsqueeze(intention_o, dim=1)  # (batch_size, 1, hidden_dim)
        h2, h_ = self.gru(h1, hidden_state)  # (batch_size, 1, hidden_dim) (1, batch_size, hidden_dim)
        h2 = torch.squeeze(h2, dim=1)  # (batch_size, hidden_dim)

        h3 = torch.cat([image_encode, lidar_encode, h2], dim=-1)
        value = self.Q_value(h3)  # (batch_size, 1)
        return value, h_


class PolicyNetwork(nn.Module):
    def __init__(self, image_shape, lidar_shape, intention_shape, action_shape, hidden_dim, graphics=True):
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
        self.action_shape = action_shape
        self.graphics = graphics
        if self.graphics:
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

        '''3.目标意图数据编码, 使用MLP, 意图数据输入维度是重点需要注意的'''
        self.intention_encode_1 = nn.Sequential(
            nn.Linear(intention_shape, hidden_dim),
            nn.ReLU())  # batch_size, object_num, hidden_dim
        self.intention_encode_2 = nn.Sequential(
            nn.Linear(intention_shape, hidden_dim),
            nn.ReLU())  # batch_size, object_num, hidden_dim

        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        '''3.策略生成'''
        self.action_probs_linear = nn.Sequential(
            nn.Linear(3 * hidden_dim, action_shape),
            nn.Softmax(dim=-1)
        )

    def forward(self, image, lidar, intention, hidden_state):
        """
        :param hidden_state:
        :param image: (batch_size, 3, 84, 84)
        :param lidar: (batch_size, lidar_shape)
        :return:
        """
        image_encode = self.conv(image)  # (batch_size, hidden_dim)
        lidar_encode = self.lidar_encode(lidar)  # (batch_size, hidden_dim)

        intention_h = self.intention_encode_1(intention)  # batch_size, object_num, hidden_dim
        intention_e = self.intention_encode_2(intention)  # batch_size, object_num, hidden_dim
        intention_a = F.softmax(intention_e, dim=1)  # batch_size, object_num, hidden_dim
        intention_o = torch.sum(intention_a * intention_h, dim=1)  # batch_size, hidden_dim
        h1 = torch.unsqueeze(intention_o, dim=1)  # (batch_size, 1, hidden_dim)
        h2, h_ = self.gru(h1, hidden_state)  # (batch_size, 1, hidden_dim) (1, batch_size, hidden_dim)
        h2 = torch.squeeze(h2, dim=1)  # (batch_size, hidden_dim)

        h3 = torch.cat([image_encode, lidar_encode, h2], dim=-1)

        action_probs = self.action_probs_linear(h3)
        pi = torch.distributions.Categorical(action_probs)
        return pi, action_probs, h_

    def get_action(self, image, lidar, intention, hidden_state, deterministic=False):
        """
        生成动作
        :param deterministic:
        :param hidden_state: # (1, batch_size, hidden_dim)
        :param image:
        :param lidar:
        :return: action_onehot, action_log_probs
        """
        pi, action_probs, h_ = self.forward(image, lidar, intention, hidden_state)
        if deterministic:
            action_index = torch.argmax(action_probs, dim=1, keepdim=False)  # [index]
        else:
            action_index = pi.sample()  # [index]
        action_onehot = torch.nn.functional.one_hot(action_index, num_classes=self.action_shape)  # [[0, 0, 1]]
        action_log_probs = pi.log_prob(action_index).cpu().detach().numpy()[0]  # [log_probs]
        action_onehot = action_onehot.cpu().detach().numpy()[0]  # [0, 0, 1]
        h_ = torch.squeeze(h_, dim=0).cpu().detach().numpy()[0] # hidden_dim
        return action_onehot, action_log_probs, h_


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
                 max_episode_num=50000,
                 resume=False,
                 hidden_dim=128):

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
        self.hidden_dim = hidden_dim

        self.epsilon = DecayedValue(scheduletype="LINEAR",
                                    initial_value=self.parameters["epsilon"],
                                    min_value=0.1,
                                    max_step=max_episode_num)

        self.learning_rate = DecayedValue(scheduletype="LINEAR",
                                          initial_value=self.parameters["learning_rate"],
                                          min_value=1e-10,
                                          max_step=max_episode_num)
        self.beta = DecayedValue(scheduletype="LINEAR",
                                 initial_value=self.parameters["beta"],
                                 min_value=1e-5,
                                 max_step=max_episode_num)

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
                                         intention_shape=5,
                                         action_shape=self.group_act_shape_list[group_index],
                                         hidden_dim=hidden_dim,
                                         graphics=True).to(device)
                           for group_index in range(self.group_num)]
            self.critics = [ValueNetwork(image_shape=(3, 84, 84),
                                         lidar_shape=self.group_obs_shape_list[group_index][1][0],
                                         intention_shape=5,
                                         hidden_dim=hidden_dim,
                                         graphics=True).to(device) for group_index in range(self.group_num)]
            self.optimizers = [
                optim.Adam(
                    params=list(self.actors[group_index].parameters()) + list(self.critics[group_index].parameters()))
                for group_index in range(self.group_num)]
        else:
            self.actors = [PolicyNetwork(image_shape=(3, 84, 84),
                                         lidar_shape=self.obs_shape_list[agent_index][1][0],
                                         intention_shape=5,
                                         action_shape=self.act_space_list[agent_index],
                                         hidden_dim=hidden_dim,
                                         graphics=True).to(device)
                           for agent_index in range(self.agent_num)]
            self.critics = [ValueNetwork(image_shape=(3, 84, 84),
                                         lidar_shape=self.obs_shape_list[agent_index][1][0],
                                         intention_shape=5,
                                         hidden_dim=hidden_dim,
                                         graphics=True).to(device) for agent_index in range(self.agent_num)]
            self.optimizers = [
                optim.Adam(
                    params=list(self.actors[agent_index].parameters()) + list(self.critics[agent_index].parameters()))
                for agent_index in range(self.agent_num)]

        self.hidden_state_actor_n = [np.zeros(hidden_dim) for i in range(self.agent_num)]
        self.hidden_state_critic_n = [np.zeros(hidden_dim) for i in range(self.agent_num)]
        self.buffers = [{'image': [],
                         'lidar': [],
                         'intention': [],
                         'hidden_state_actor':[],
                         'hidden_state_critic':[],
                         'action': [],
                         'old_action_log_probs': [],
                         'reward': [],
                         'done': [],
                         'old_value': [],
                         'value_next': []
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

    def action(self, obs_n, evaluation=False, global_step=0):  # 所有智能体obs in,所有action out
        action_n = []
        for i, obs in enumerate(obs_n):
            image = torch.as_tensor([obs[0]], dtype=torch.float32, device=device)
            lidar = torch.as_tensor([obs[1]], dtype=torch.float32, device=device)
            intention = torch.tensor([np.reshape(np.asarray(obs[2]), [8, 5])], dtype=torch.float32, device=device)
            hidden_state_actor = torch.unsqueeze(torch.as_tensor([self.hidden_state_actor_n[i]], dtype=torch.float32, device=device), dim=0)
            hidden_state_critic = torch.unsqueeze(torch.as_tensor([self.hidden_state_critic_n[i]], dtype=torch.float32, device=device), dim=0)
            with torch.no_grad():
                if self.share_parameters:
                    action_onehot, action_log_probs, h_a = self.actors[self.agent_group_index[i]].get_action(image=image,
                                                                                                             lidar=lidar,
                                                                                                             intention=intention,
                                                                                                             hidden_state=hidden_state_actor,
                                                                                                             deterministic=evaluation)
                    old_value, h_c = self.critics[self.agent_group_index[i]](image, lidar, intention, hidden_state_critic)
                    h_c = torch.squeeze(h_c, dim=0).cpu().detach().numpy()[0]  # hidden_dim
                else:
                    action_onehot, action_log_probs, h_a = self.actors[i].get_action(image=image,
                                                                                     lidar=lidar,
                                                                                     intention=intention,
                                                                                     hidden_state=hidden_state_actor,
                                                                                     deterministic=evaluation)
                    old_value, h_c = self.critics[i](image, lidar, intention, hidden_state_critic)
                    h_c = torch.squeeze(h_c, dim=0).cpu().detach().numpy()[0]  # hidden_dim
            action_n.append(action_onehot)
            self.buffers[i]['old_action_log_probs'].append(action_log_probs)
            self.buffers[i]['old_value'].append(old_value.cpu().detach().numpy()[0])
            self.buffers[i]['hidden_state_actor'].append(self.hidden_state_actor_n[i])
            self.buffers[i]['hidden_state_critic'].append(self.hidden_state_critic_n[i])
            self.hidden_state_actor_n[i] = h_a
            self.hidden_state_critic_n[i] = h_c

        return action_n

    # 存储transition
    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n, info_n):
        for i in range(self.agent_num):
            self.buffers[i]['image'].append(obs_n[i][0])
            self.buffers[i]['lidar'].append(obs_n[i][1])
            self.buffers[i]['intention'].append(np.reshape(np.asarray(obs_n[i][2]), [8, 5]))
            self.buffers[i]['action'].append(act_n[i])
            self.buffers[i]['reward'].append(rew_n[i])
            self.buffers[i]['done'].append(float(done_n[i]))

            image = torch.as_tensor([new_obs_n[i][0]], dtype=torch.float32, device=device)
            lidar = torch.as_tensor([new_obs_n[i][1]], dtype=torch.float32, device=device)
            intention = torch.tensor([np.reshape(np.asarray(new_obs_n[i][2]), [8, 5])], dtype=torch.float32, device=device)
            hidden_state_critic = torch.unsqueeze(torch.as_tensor([self.hidden_state_critic_n[i]], dtype=torch.float32, device=device), dim=0)
            with torch.no_grad():
                if self.share_parameters:
                    value_next, _ = self.critics[self.agent_group_index[i]](image, lidar, intention, hidden_state_critic)
                else:
                    value_next, _ = self.critics[i](image, lidar, intention, hidden_state_critic)
            self.buffers[i]['value_next'] = list(value_next.cpu().detach().numpy()[0])

            if float(done_n[i]):
                self.reset_gru_hidden(i)

    def reset_gru_hidden(self, agent_index):
        self.hidden_state_actor_n[agent_index] = np.zeros(self.hidden_dim)
        self.hidden_state_critic_n[agent_index] = np.zeros(self.hidden_dim)

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

    def can_update(self):
        can_up = []
        for i in range(self.agent_num):
            if len(self.buffers[i]['done']) == self.parameters["buffer_size"]:
                can_up.append(True)
            else:
                can_up.append(False)
        return all(can_up)

    def update(self, train_step):
        EPSILON = self.epsilon.get_value(train_step)
        LEARNING_RATE = self.learning_rate.get_value(train_step)
        BETA = self.beta.get_value(train_step)
        """更新学习率"""
        if self.share_parameters:
            for group_index in range(self.group_num):
                for param_group in self.optimizers[group_index].param_groups:
                    param_group["lr"] = LEARNING_RATE
        else:
            for agent_index in range(self.agent_num):
                for param_group in self.optimizers[agent_index].param_groups:
                    param_group["lr"] = LEARNING_RATE
        summaries = {
            'Loss/Policy_loss': [0 for _ in range(self.agent_num)],
            'Loss/Value_loss': [0 for _ in range(self.agent_num)],
            'Policy/Entropy': [0 for _ in range(self.agent_num)],
            'Policy/Epsilon': [EPSILON for _ in range(self.agent_num)],
            'Policy/Learning_rate': [LEARNING_RATE for _ in range(self.agent_num)],
            'Policy/Beta': [BETA for _ in range(self.agent_num)],
        }

        image_n = []
        lidar_n = []
        intention_n = []
        action_n = []
        old_action_log_probs_n = []
        discount_reward_n = []
        old_value_n = []
        advantage_n = []
        hidden_state_actor_n = []
        hidden_state_critic_n = []
        for agent_index in range(self.agent_num):
            """1.获取数据"""
            image = torch.tensor(self.buffers[agent_index]['image'], dtype=torch.float32, device=device)
            lidar = torch.tensor(self.buffers[agent_index]['lidar'], dtype=torch.float32, device=device)
            intention = torch.tensor(self.buffers[agent_index]['intention'], dtype=torch.float32, device=device)
            action = torch.tensor(self.buffers[agent_index]['action'], dtype=torch.float32, device=device)
            old_action_log_probs = torch.tensor(self.buffers[agent_index]['old_action_log_probs'], dtype=torch.float32,
                                                device=device)
            old_value = torch.tensor(self.buffers[agent_index]['old_value'], dtype=torch.float32, device=device)
            hidden_state_actor = torch.tensor(self.buffers[agent_index]['hidden_state_actor'], dtype=torch.float32, device=device)
            hidden_state_critic = torch.tensor(self.buffers[agent_index]['hidden_state_critic'], dtype=torch.float32, device=device)

            """2.计算discount_reward"""
            discount_reward = self.discount_rewards(r=self.buffers[agent_index]['reward'],
                                                    done=self.buffers[agent_index]['done'],
                                                    gamma=self.parameters['gamma'],
                                                    value_next=self.buffers[agent_index]['value_next'][0])
            discount_reward = torch.tensor(discount_reward, dtype=torch.float32, device=device)

            if self.parameters['use_gae_adv']:
                advantage = self.get_gae(rewards=self.buffers[agent_index]['reward'],
                                         done=self.buffers[agent_index]['done'],
                                         value_estimates=torch.squeeze(old_value, dim=1).cpu().detach().numpy(),
                                         value_next=self.buffers[agent_index]['value_next'][0],
                                         gamma=self.parameters['gamma'],
                                         lambd=self.parameters['lambda'])
                advantage = torch.tensor(advantage, dtype=torch.float32, device=device)
            else:
                advantage = discount_reward.detach() - torch.squeeze(old_value, dim=-1).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            image_n.append(image)
            lidar_n.append(lidar)
            intention_n.append(intention)
            action_n.append(action)
            old_action_log_probs_n.append(old_action_log_probs)
            discount_reward_n.append(discount_reward)
            old_value_n.append(old_value)
            advantage_n.append(advantage)
            hidden_state_actor_n.append(hidden_state_actor)
            hidden_state_critic_n.append(hidden_state_critic)

        for _ in range(self.parameters['num_epoch']):
            batch_number = self.parameters['buffer_size'] // self.parameters['batch_size']
            for i in range(batch_number):
                index = torch.randint(0, self.parameters['buffer_size'], (self.parameters['batch_size'],),
                                      device=device)
                if self.share_parameters:
                    policys_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
                    critics_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
                    entropys_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
                    for agent_index in range(self.agent_num):
                        image_batch = torch.index_select(image_n[agent_index], 0, index)
                        lidar_batch = torch.index_select(lidar_n[agent_index], 0, index)
                        intention_batch = torch.index_select(intention_n[agent_index], 0, index)
                        action_batch = torch.index_select(action_n[agent_index], 0, index)
                        old_action_log_probs_batch = torch.index_select(old_action_log_probs_n[agent_index], 0, index)
                        discount_reward_batch = torch.index_select(discount_reward_n[agent_index], 0, index)
                        old_value_batch = torch.index_select(old_value_n[agent_index], 0, index)
                        advantage_batch = torch.index_select(advantage_n[agent_index], 0, index)
                        hidden_state_a_batch = torch.unsqueeze(torch.index_select(hidden_state_actor_n[agent_index], 0, index), dim=0)
                        hidden_state_c_batch = torch.unsqueeze(torch.index_select(hidden_state_critic_n[agent_index], 0, index), dim=0)

                        value, _ = self.critics[self.agent_group_index[agent_index]](image_batch, lidar_batch, intention_batch, hidden_state_c_batch)
                        pi, _, _ = self.actors[self.agent_group_index[agent_index]](image_batch, lidar_batch, intention_batch, hidden_state_a_batch)
                        # actor_loss
                        action_log_probs = pi.log_prob(torch.argmax(action_batch, dim=-1, keepdim=False))
                        r_theta = torch.exp(action_log_probs - old_action_log_probs_batch)
                        p_opt_a = r_theta * advantage_batch
                        p_opt_b = torch.clamp(r_theta, 1.0 - EPSILON, 1.0 + EPSILON) * advantage_batch
                        policy_loss = -1 * torch.mean(torch.min(p_opt_a, p_opt_b))
                        # entropy
                        entropy_loss = torch.mean(pi.entropy())
                        # critic_loss
                        # critic_loss = nn.MSELoss()(torch.squeeze(value, dim=1), discount_reward.detach())
                        td_error = discount_reward_batch.detach() - torch.squeeze(value, dim=1)
                        value_clip = torch.squeeze(old_value_batch, dim=1) + torch.clamp(
                            torch.squeeze(value, dim=1) - torch.squeeze(old_value_batch, dim=1), -1 * EPSILON, EPSILON)
                        td_error_clip = discount_reward_batch.detach() - value_clip
                        td_square = torch.maximum(torch.square(td_error), torch.square(td_error_clip))
                        critic_loss = torch.mean(td_square)

                        policys_loss[self.agent_group_index[agent_index]] += policy_loss
                        critics_loss[self.agent_group_index[agent_index]] += critic_loss
                        entropys_loss[self.agent_group_index[agent_index]] += entropy_loss

                        summaries['Loss/Policy_loss'][agent_index] += policy_loss * (
                                1.0 / (batch_number * self.parameters['num_epoch']))
                        summaries['Loss/Value_loss'][agent_index] += critic_loss * (
                                1.0 / (batch_number * self.parameters['num_epoch']))
                        summaries['Policy/Entropy'][agent_index] += entropy_loss * (
                                1.0 / (batch_number * self.parameters['num_epoch']))

                    for group_index in range(self.group_num):
                        # total loss
                        loss = policys_loss[group_index] + 0.5 * critics_loss[group_index] - BETA * entropys_loss[
                            group_index]
                        self.optimizers[group_index].zero_grad()
                        loss.backward()
                        self.optimizers[group_index].step()
                else:
                    # policys_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
                    # critics_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
                    # entropys_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
                    for agent_index in range(self.agent_num):
                        image_batch = torch.index_select(image_n[agent_index], 0, index)
                        lidar_batch = torch.index_select(lidar_n[agent_index], 0, index)
                        intention_batch = torch.index_select(intention_n[agent_index], 0, index)
                        action_batch = torch.index_select(action_n[agent_index], 0, index)
                        old_action_log_probs_batch = torch.index_select(old_action_log_probs_n[agent_index], 0, index)
                        discount_reward_batch = torch.index_select(discount_reward_n[agent_index], 0, index)
                        old_value_batch = torch.index_select(old_value_n[agent_index], 0, index)
                        advantage_batch = torch.index_select(advantage_n[agent_index], 0, index)
                        hidden_state_a_batch = torch.unsqueeze(torch.index_select(hidden_state_actor_n[agent_index], 0, index), dim=0)
                        hidden_state_c_batch = torch.unsqueeze(torch.index_select(hidden_state_critic_n[agent_index], 0, index), dim=0)

                        value, _ = self.critics[agent_index](image_batch, lidar_batch, intention_batch, hidden_state_c_batch)
                        pi, _, _ = self.actors[agent_index](image_batch, lidar_batch, intention_batch, hidden_state_a_batch)
                        # actor_loss
                        action_log_probs = pi.log_prob(torch.argmax(action_batch, dim=-1, keepdim=False))
                        r_theta = torch.exp(action_log_probs - old_action_log_probs_batch)
                        p_opt_a = r_theta * advantage_batch
                        p_opt_b = torch.clamp(r_theta, 1.0 - EPSILON, 1.0 + EPSILON) * advantage_batch
                        policy_loss = -1 * torch.mean(torch.min(p_opt_a, p_opt_b))
                        # entropy
                        entropy_loss = torch.mean(pi.entropy())
                        # critic_loss
                        # critic_loss = nn.MSELoss()(torch.squeeze(value, dim=1), discount_reward.detach())
                        td_error = discount_reward_batch.detach() - torch.squeeze(value, dim=1)
                        value_clip = torch.squeeze(old_value_batch, dim=1) + torch.clamp(
                            torch.squeeze(value, dim=1) - torch.squeeze(old_value_batch, dim=1), -1 * EPSILON, EPSILON)
                        td_error_clip = discount_reward_batch.detach() - value_clip
                        td_square = torch.maximum(torch.square(td_error), torch.square(td_error_clip))
                        critic_loss = torch.mean(td_square)

                        # total loss
                        loss = policy_loss + 0.5 * critic_loss - BETA * entropy_loss
                        self.optimizers[agent_index].zero_grad()
                        loss.backward()
                        self.optimizers[agent_index].step()

                        summaries['Loss/Policy_loss'][agent_index] += policy_loss * (
                                1.0 / (batch_number * self.parameters['num_epoch']))
                        summaries['Loss/Value_loss'][agent_index] += critic_loss * (
                                1.0 / (batch_number * self.parameters['num_epoch']))
                        summaries['Policy/Entropy'][agent_index] += entropy_loss * (
                                1.0 / (batch_number * self.parameters['num_epoch']))

        # 每一轮训练结束，清空本轮所有的buffer
        for buffer in self.buffers:
            for key in buffer.keys():
                buffer[key].clear()

        for i in range(self.agent_num):
            for key in summaries.keys():
                self.summary_writers[i].add_scalar(key,
                                                   summaries[key][i],
                                                   global_step=train_step)
            self.summary_writers[i].flush()

    def discount_rewards(self, r, done, gamma=0.99, value_next=0.0):
        """
        Computes discounted sum of future rewards for use in updating value estimate.
        :param r: List of rewards.
        :param gamma: Discount factor.
        :param value_next: T+1 value estimate for returns calculation.
        :return: discounted sum of future rewards as list.
        """
        discounted_r = [0 for _ in range(len(r))]
        running_add = value_next
        for t in reversed(range(0, len(r))):
            running_add = running_add * gamma * (1.0 - done[t]) + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def get_gae(self, rewards, done, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
        """
        Computes generalized advantage estimate for use in updating policy.
        :param rewards: list of rewards for time-steps t to T.
        :param value_next: Value estimate for time-step T+1.
        :param value_estimates: list of value estimates for time-steps t to T.
        :param gamma: Discount factor.
        :param lambd: GAE weighing factor.
        :return: list of advantage estimates for time-steps t to T.
        """
        value_estimates = np.append(value_estimates, value_next)
        delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
        advantage = self.discount_rewards(r=delta_t, done=done, gamma=gamma * lambd)
        return advantage
