import sys
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import logging
import matplotlib.pyplot as plt
from mlagents_envs.YOLOv5 import yolov5
import cv2
from PIL import Image


class Unity_Env(object):
    """
    Unity_Env类基于Unity官方开源的mlagents项目, 形成多智能体环境封装类。
    该类目前支持情况:
    - 支持mlagents版本:17
    - 环境中每个智能体的behavier_name需要不一样或者team_id不一样
    - 支持连续和离散动作，多个智能体的动作类型必须唯一，即要么都是连续的，要么都是离散的
    - 支持图像(image)+激光雷达(lidar)+向量化observation(vector_obs), Unity默认输出的顺序是图像(如84,84,3)->激光->向量
    - 如没有特殊设定, 每个智能体会获得三维list:[image, lidar, vector_obs]
    - 对于图像数据可以引入目标识别模型，将图像识别的结果也加入观测信息中：[image, lidar, vector_obs, detector_img, detector_info]
    - 由于在外部加入目标识别模型会造成训练速度大幅下降, 导致无法训练, 因此一般在Unity中进行目标检测, 检测结果在vector_obs中输出
    - 目前不支持使用外部目标识别模型参与训练, 仅支持Unity内部目标检测
    """

    def __init__(self,
                 file_name="",
                 no_graphics=False,
                 time_scale=1,
                 worker_id=0,
                 camera_detect_with_YOLOv5=False):
        """
        :param file_name:
        :param no_graphics:
        :param time_scale:
        :param worker_id:
        :param camera_detect_with_YOLOv5:
        """
        self.engine_configuration_channel = EngineConfigurationChannel()
        if file_name == "":
            self.env = UnityEnvironment(worker_id=worker_id,
                                        side_channels=[self.engine_configuration_channel])
        else:
            self.env = UnityEnvironment(file_name=file_name,
                                        worker_id=worker_id,
                                        no_graphics=no_graphics,
                                        side_channels=[self.engine_configuration_channel])

        self.engine_configuration_channel.set_configuration_parameters(
            width=1920,
            height=1080,
            # quality_level = 5, #1-5
            time_scale=time_scale  # 1-100, 10执行一轮的时间约为10秒，20执行一轮的时间约为5秒。
            # target_frame_rate = 60, #1-60
            # capture_frame_rate = 60 #default 60
        )
        self.env.reset()

        self.camera_detect_with_YOLOv5 = camera_detect_with_YOLOv5
        if self.camera_detect_with_YOLOv5:
            self.camera_detector = yolov5(path=os.path.abspath(os.path.dirname(__file__)) + "/yolov5n_best.pt")

        # 获得所有的behavior_name,
        # 名称结构如：DefenderCarAgent?team=0， 
        # 因此在制作环境时应该保持每个智能体对应一个名称
        self.agent_names = self.env.behavior_specs.keys()
        self.n = len(self.agent_names)

        # state(observation_specs)主要分为三个部分,固定的Unity输出顺序为:【图像、雷达、向量】
        self.observation_space = []
        for agent_index, behavior_name in enumerate(self.agent_names):
            obs = []
            for o in self.env.behavior_specs.get(behavior_name).observation_specs:
                obs.append(o.shape)
            if self.camera_detect_with_YOLOv5:
                obs.append((10, 6))
            self.observation_space.append(obs)

        # 是否是连续动作，连续动作：True，离散动作：False
        self.action_type = [self.env.behavior_specs.get(behavior_name).action_spec.is_continuous() for behavior_name in
                            self.agent_names]

        self.action_space = []
        for agent_index, action_type in enumerate(self.action_type):
            if action_type:
                self.action_space.append(
                    self.env.behavior_specs.get(
                        list(self.agent_names)[agent_index]).action_spec.continuous_size)
            else:
                self.action_space.append(
                    self.env.behavior_specs.get(
                        list(self.agent_names)[agent_index]).action_spec.discrete_branches[0])

    def reset(self):
        self.env.reset()
        cur_state = []
        for behavior_name in self.agent_names:
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            obs = []
            for o_index, o in enumerate(DecisionSteps.obs):
                if o_index == 0:
                    obs.append(np.transpose(o[0], (2, 0, 1)))
                else:
                    obs.append(o[0])
            if self.camera_detect_with_YOLOv5:
                detector_info, detector_img = self.camera_detector.imageDetectorfromFrame(
                    np.transpose(obs[0], (1, 2, 0)))
                obs.append(detector_img)
                if detector_info.shape[0] < 10:
                    obs.append(np.concatenate((detector_info, np.zeros((10 - detector_info.shape[0], 6))), axis=0))
                else:
                    obs.append(detector_info[:10])
            cur_state.append(obs)
        return cur_state  # agent_number x [image, lidar_vector, obs_vector, detector_img, detector_info]

    def step(self, actions):
        """
        :param actions: 所有智能体动作的集合, 连续动作时集合中每个智能体动作维度为初始动作维度(如[0.5, 0.6]), 离散动作为one-hot编码(如[1, 0, 0, 0, 0, 0])
        """
        next_state = []
        reward = []
        done = []
        info = []
        # t1 = time.time()
        for behavior_name_index, behavior_name in enumerate(self.agent_names):
            action = ActionTuple()
            if self.action_type[behavior_name_index]:
                action.add_continuous(np.asarray(actions[behavior_name_index]).reshape((1, -1)))
            else:
                action.add_discrete(np.asarray([[actions[behavior_name_index].argmax(-1)]]))
            self.env.set_actions(behavior_name=behavior_name, action=action)
        self.env.step()
        # t2 = time.time()
        # print("step set action time:" + str(t2-t1))

        # t1 = time.time()
        for i, behavior_name in enumerate(self.agent_names):
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            if len(TerminalSteps.reward) == 0:
                obs = []
                for o_index, o in enumerate(DecisionSteps.obs):
                    if o_index == 0:
                        obs.append(np.transpose(o[0], (2, 0, 1)))
                    else:
                        obs.append(o[0])
                    if self.camera_detect_with_YOLOv5:
                        detector_info, detector_img = self.camera_detector.imageDetectorfromFrame(
                            np.transpose(obs[0], (1, 2, 0)))
                        obs.append(detector_img)
                        if detector_info.shape[0] < 10:
                            obs.append(
                                np.concatenate((detector_info, np.zeros((10 - detector_info.shape[0], 6))), axis=0))
                        else:
                            obs.append(detector_info[:10])
                next_state.append(obs)
                reward.append(DecisionSteps.reward[0] + DecisionSteps.group_reward[0])
                done.append(False)
                info.append(False)
            else:
                obs = []
                for o_index, o in enumerate(TerminalSteps.obs):
                    if o_index == 0:
                        obs.append(np.transpose(o[0], (2, 0, 1)))
                    else:
                        obs.append(o[0])
                    if self.camera_detect_with_YOLOv5:
                        detector_info, detector_img = self.camera_detector.imageDetectorfromFrame(
                            np.transpose(obs[0], (1, 2, 0)))
                        obs.append(detector_img)
                        if detector_info.shape[0] < 10:
                            obs.append(
                                np.concatenate((detector_info, np.zeros((10 - detector_info.shape[0], 6))), axis=0))
                        else:
                            obs.append(detector_info[:10])
                next_state.append(obs)
                reward.append(TerminalSteps.reward[0] + TerminalSteps.group_reward[0])
                reachmaxstep = TerminalSteps.interrupted
                done.append(True)
                info.append(reachmaxstep[0])
        # t2 = time.time()
        # print("step get obs time:" + str(t2-t1))
        return next_state, reward, done, info

    def close(self):
        self.env.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = Unity_Env(
        file_name=r"C:\\Users\\liyang\Desktop\\MARLS_Unity\\Unity_Envs\\USV-AG-Scenario2\\USV-AG-Scenario2.exe",
        worker_id=2,
        no_graphics=False,
        time_scale=20,
        camera_detect_with_YOLOv5=False)
    # env = Unity_Env(time_scale=1, camera_detect_with_YOLOv5=False)
    print(list(env.agent_names))
    print(env.n)
    print(env.observation_space)
    print(env.action_space)
    print(env.action_type)

    # plt.ion()
    cameras = plt.figure()  # 建立图窗
    for j in range(10):

        cur_sate = env.reset()

        for i in range(500):
            t3 = time.time()

            actions = []
            for action_dim in env.action_space:
                actions.append(np.random.uniform(-1, 1, action_dim))
            next_state, reward, done, info = env.step(actions)


            '''获取agent0的视觉传感器图像'''
            # image = np.transpose(next_state[0][0], (1, 2, 0))

            '''plt显示观测图像'''
            # image shape (84, 84, 1)
            # plt.imshow(image.squeeze(), cmap='gray')  # 绘图展现
            # image shape (84, 84, 3)
            # plt.imshow(image)  # 绘图展现
            # plt.pause(0.1)
            # cameras.clf()
            # plt.show()

            '''cv2显示观测图像'''
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("USV", image)
            # cv2.waitKey(1)

            '''cv2显示识别后的图像'''
            # image_detector = next_state[0][-2]
            # image = cv2.cvtColor(image_detector, cv2.COLOR_RGB2BGR)
            # cv2.imshow("USV", image)
            # cv2.waitKey(1)

            '''备用cv2使用方法'''
            # cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
            # cv2.imwrite("./images/" + str(j) + "-" + str(i) + ".jpg", image)
            # # detector_info, detector_img = env.camera_detector.imageDetectorfromFrame(image, floatorint=True)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # # image = camera_detector.imageDetectorfromFile(filename="./images/" + str(j) + "-" + str(i) + ".jpg")
            # # cv2.imwrite("./images/"+str(j)+"-"+str(i)+".jpg", image)

            print("step:{}, reward:{}, done:{}, info:{}".format(i, reward, done, info))
            if all(done):
                break

    env.close()
    cv2.destroyAllWindows()
    # plt.ioff()
