import argparse
import json
import logging
import numpy as np
import time, os
import agents
from mlagents_envs.Unity_Env import Unity_Env
from model_parameters import parameters
import signal


def handler_stop(a, b):
    print("关闭程序成功！！！")
    exit()


def train(train_parameters, model_parameters, model_path, log_path):
    '''
    :param train_parameters: 训练参数
    :param model_parameters: 模型超参数
    :param model_path: 模型保存位置
    :param log_path: 训练数据保存位置
    :return:
    '''

    # 初始化环境
    if train_parameters["env_run_type"] == 'exe':
        logging.basicConfig(level=logging.INFO)
        env = Unity_Env(file_name="./Unity_Envs/" +
                                  train_parameters["env_name"] + "/" +
                                  train_parameters["env_name"],
                        no_graphics=train_parameters["no_graphics"],
                        worker_id=train_parameters["env_worker_id"],
                        time_scale=20)
    else:
        logging.basicConfig(level=logging.INFO)
        env = Unity_Env(time_scale=20)

    # 初始化Multi-Agents model
    train_agents = agents.load(train_parameters['train_algorithm'] + ".py").Agent(
        name=train_parameters['env_name'],
        obs_shape=env.observation_space,
        act_type=env.action_type,
        act_space=env.action_space,
        agent_num=env.n,
        group_num=train_parameters['group_num'],
        agent_group_index=train_parameters['agent_group_index'],
        share_parameters=train_parameters['share_parameters'],
        parameters=model_parameters,
        model_path=model_path,
        log_path=log_path,
        create_summary_writer=True,
        max_episode_num=train_parameters['num_episodes'],
        resume=train_parameters['resume'])

    print('Starting training...')
    episode = 0
    epoch = train_agents.log_info_json["epoch"]
    train_step = train_agents.log_info_json["train_step"]
    log_episode = train_agents.log_info_json["log_episode"]
    print('******************本次训练从epoch=' + str(epoch) + " train_step=" + str(
        train_step) + "处开始***************************")
    t_start = time.time()

    try:
        while episode < train_parameters["num_episodes"]:
            obs_n = env.reset()
            episode_steps = 0
            episode_rewards = [0.0]  # sum of rewards for all agents
            group_rewards = [0.0 for i in range(train_parameters["group_num"])]
            agent_rewards = [0.0 for _ in range(env.n)]  # individual agent reward

            if train_parameters["train_algorithm"] == "SAC" or \
                    train_parameters["train_algorithm"] == "SAC_DISCRETE" or \
                    train_parameters["train_algorithm"] == "PPO" or \
                    train_parameters["train_algorithm"] == "PPO_DISCRETE" or \
                    train_parameters["train_algorithm"] == "DGN" or \
                    train_parameters["train_algorithm"] == "DQN" or \
                    train_parameters["train_algorithm"] == "VDN" or \
                    train_parameters["train_algorithm"] == "QMIX" or \
                    train_parameters["train_algorithm"] == "INTENTION_DQN":
                pass
            else:
                for action_noise in train_agents.action_noises:
                    action_noise.reset()

            new_obs_n, done_n = [], []
            while episode_steps < train_parameters["max_episode_len"]:
                # get action
                action_n = train_agents.action(obs_n, evaluation=False, global_step=epoch)
                # environment step
                # print(action_n)
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                done = all(done_n)
                # collect experience
                train_agents.experience(obs_n, action_n, rew_n, new_obs_n, done_n, info_n)

                # 记录reward数据
                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i] += rew
                # 记录每个组的数据
                for agent_index, group_index in enumerate(train_parameters["agent_group_index"]):
                    group_rewards[group_index] += rew_n[agent_index]

                if train_parameters["train_algorithm"] == "PPO" or train_parameters["train_algorithm"] == "PPO_DISCRETE":
                    '''PPO需要每一个episode后再开始训练'''
                    if train_agents.can_update():
                        train_agents.update(log_episode + episode)
                        train_step += 1
                else:
                    # update all trainers
                    if epoch % train_parameters["train_frequency"] == 0:
                        if train_agents.can_update():
                            train_agents.update(train_step)
                            train_step += 1

                if done:
                    break

                obs_n = new_obs_n
                episode_steps += 1
                epoch += 1

            # if train_parameters["train_algorithm"] == "PPO" or train_parameters["train_algorithm"] == "PPO_DISCRETE":
            #     train_agents.update(train_step)
            #     train_step += 1

            if episode % train_parameters["print_frequency"] == 0:
                print(
                    f"Episode: {log_episode + episode:3d}\t"
                    f"Episode Steps: {episode_steps: 2d}\t"
                    f"Epoch: {epoch: 3d}\t"
                    f"Train Steps: {train_step: 3d}\t"
                    f"Time: {time.time() - t_start: 6.3f}\t"
                    f"Reward: {agent_rewards}"
                )
                t_start = time.time()

            episode += 1

            for i, summary_writer in enumerate(train_agents.summary_writers):
                summary_writer.add_scalar('A_Main/total_reward', episode_rewards[-1], log_episode + episode)
                summary_writer.add_scalar('A_Main/Agent_reward', agent_rewards[i], log_episode + episode)
                summary_writer.add_scalar('A_Main/group_reward',
                                          group_rewards[train_parameters["agent_group_index"][i]],
                                          log_episode + episode)
                summary_writer.add_scalar('A_Main/episode_steps', episode_steps, log_episode + episode)
                summary_writer.flush()

            if episode != 0 and episode % train_parameters["save_frequency"] == 0:
                # 保存模型参数
                train_agents.save_model()

    except KeyboardInterrupt:
        print("人为取消训练。。。。。")
        # 保存log_info_json信息
        train_agents.log_info_json["epoch"] = epoch
        train_agents.log_info_json["train_step"] = train_step
        train_agents.log_info_json["log_episode"] = log_episode + episode
        print("保存断点中。。。。。")
        time.sleep(0.5)
        with open(log_path + '/log_info.txt', "w") as fp:
            fp.write(json.dumps(train_agents.log_info_json))
            fp.close()
        # 保存模型参数
        print("保存模型中。。。。。")
        time.sleep(0.5)
        train_agents.save_model()

        # 关闭summary，回收资源
        print("关闭summary中。。。。。")
        time.sleep(0.5)
        for summary_writer in train_agents.summary_writers:
            summary_writer.close()
        env.close()
        print("关闭程序成功！！！")
        exit()

    # 保存log_info_json信息
    train_agents.log_info_json["epoch"] = epoch
    train_agents.log_info_json["train_step"] = train_step
    train_agents.log_info_json["log_episode"] = log_episode + episode
    with open(log_path + '/log_info.txt', "w") as fp:
        fp.write(json.dumps(train_agents.log_info_json))
        fp.close()
    # 保存模型参数
    train_agents.save_model()

    # 关闭summary，回收资源
    for summary_writer in train_agents.summary_writers:
        summary_writer.close()
    env.close()


def inference(train_parameters, model_parameters, model_path):
    '''
    :param train_parameters:
    :param model_parameters:
    :param model_path:
    :return:
    '''

    # 初始化环境
    if train_parameters["env_run_type"] == 'exe':
        env = Unity_Env(file_name="./Unity_Envs/" +
                                  train_parameters["env_name"] + "/" +
                                  train_parameters["env_name"],
                        no_graphics=False,
                        worker_id=train_parameters["env_worker_id"],
                        time_scale=2)
    else:
        logging.basicConfig(level=logging.INFO)
        env = Unity_Env(time_scale=2)

    print('******************环境加载成功**********************')
    # 初始化MADDPGAgent
    inference_agents = agents.load(train_parameters['train_algorithm'] + ".py").Agent(
        name=train_parameters['env_name'],
        obs_shape=env.observation_space,
        act_type=env.action_type,
        act_space=env.action_space,
        agent_num=env.n,
        group_num=train_parameters['group_num'],
        agent_group_index=train_parameters['agent_group_index'],
        share_parameters=train_parameters['share_parameters'],
        parameters=model_parameters,
        model_path=model_path,
        log_path=log_path,
        create_summary_writer=False)
    inference_agents.load_actor()
    print('******************模型加载成功******************')
    print('******************Starting inference...******************')
    episode = 0
    while episode < train_parameters["num_episodes"]:
        rewards = np.zeros(env.n, dtype=np.float32)
        # t0 = time.time()
        cur_state = env.reset()
        # t1 = time.time()
        # print("env reset time:"+str(t1-t0))
        step = 0
        t0 = time.time()
        while step < train_parameters["max_episode_len"]:
            # get action
            # t1 = time.time()
            action_n = inference_agents.action(cur_state, evaluation=True)
            # t2 = time.time()
            # print("get action time:"+str(t2-t1))
            # print(action_n)
            # environment step
            # t1 = time.time()
            next_state, reward, done, _ = env.step(action_n)
            # t2 = time.time()
            # print("env step time:"+str(t2-t1))
            done = all(done)

            cur_state = next_state
            rewards += np.asarray(reward, dtype=np.float32)
            step += 1
            if done:
                if train_parameters["train_algorithm"] == "PPO_DISCRETE":
                    for agent_index in range(inference_agents.agent_num):
                        inference_agents.reset_gru_hidden(agent_index)
                break
        episode += 1

        print(
            f"Episode: {episode:3d}\t"
            f"Episode Step: {step: 2d}\t"
            f"Total Reward: {rewards}\t"
            f"Time: {time.time() - t0: 6.3f}"
        )
    env.close()


if __name__ == '__main__':
    # *******************************************1、训练参数设置*********************************************************
    parser = argparse.ArgumentParser(
        description="Multi-Agent Reinforcement Learning Training Framework for Unity Environments")
    parser.add_argument('--env-name', type=str, default='USVNavigation', help='unity环境的名称')
    parser.add_argument('--env-run-type', type=str, default='exe',
                        help='unity训练环境客户端训练还是可执行程序训练: \"exe\" or \"client\"')
    parser.add_argument('--env-worker-id', type=int, default=0,
                        help='\"exe\"环境的worker_id, 可进行多环境同时训练, 默认为0, 使用client是必须设为0!')
    parser.add_argument('--no-graphics', action="store_true", help='使用unity训练时是否打开界面')
    parser.add_argument('--group-num', type=int, default=1, help='环境中智能体 组/类别 的数量')
    parser.add_argument('--agent-group-index', nargs='+', type=int, default=[0],
                        help='环境中每个agent对应的组编号')
    parser.add_argument('--share-parameters', action="store_true", help='环境中每组智能体是否组间共享网络')
    parser.add_argument('--discrete-action', action="store_true", help='环境中是否是离散的动作')
    parser.add_argument('--graph-obs', action="store_true", help='环境中是否是图形式的观测')
    parser.add_argument('--train-algorithm', type=str, default='PPO_DISCRETE',
                        help='训练算法: 目前支持DDPG, MADDPG, SAC, PPO,且名称都需要大写')
    parser.add_argument('--train', action="store_true", help='是否训练')
    parser.add_argument('--inference', action="store_true", help='是否推断')
    parser.add_argument('--max-episode-len', type=int, default=500, help='每个episode的最大step数')
    parser.add_argument('--num-episodes', type=int, default=40000, help='episode数量')
    parser.add_argument('--save-frequency', type=int, default=50, help='模型保存频率')
    parser.add_argument('--train-frequency', type=int, default=50, help='模型训练频率, 1表示每个step调用训练函数一次')
    parser.add_argument('--print-frequency', type=int, default=1, help='训练数据打印输出频率, 100表示每100轮打印一次')
    parser.add_argument('--resume', action="store_true", help='是否按照上一次的训练结果，继续训练')
    args = parser.parse_args()
    train_parameters = vars(args)

    # *******************************************2、打印Logo*********************************************************
    print(
        """
                                                                                ▓             
   ▓▓*   ▓▓▓     ▓▓▓    ▓▓▓▓▓*  ▓▓    *▓▓▓▓*          ▓▓▓    ▓▓         ▓▓▓ ▓▓         
  *▓▓▓   ▓▓▓    ▓▓▓▓   *▓▓▓▓▓▓  ▓▓    ▓▓▓▓▓▓          ▓▓▓    ▓▓          *  ▓▓         
  ▓▓▓▓  ▓▓▓▓    ▓▓▓▓*  *▓▓  ▓▓▓ ▓▓   *▓▓  *           ▓▓▓    ▓▓  ▓ *▓*   ▓ ▓▓▓▓*▓    ▓ 
  ▓▓▓▓* ▓▓▓▓   *▓▓ ▓▓  *▓▓  ▓▓▓ ▓▓   *▓▓▓             ▓▓▓    ▓▓ ▓▓▓▓▓▓▓ ▓▓*▓▓▓▓▓▓▓  ▓▓ 
  ▓▓*▓▓ ▓▓▓▓*  ▓▓* ▓▓  *▓▓*▓▓▓* ▓▓    ▓▓▓▓▓           ▓▓▓    ▓▓ ▓▓▓ ▓▓▓ *▓* ▓▓* ▓▓  ▓▓ 
  ▓▓ ▓▓*▓*▓▓▓  ▓▓  ▓▓* *▓▓▓▓▓▓  ▓▓     ▓▓▓▓▓          ▓▓▓    ▓▓ ▓▓▓  ▓▓ *▓* ▓▓  ▓▓ *▓▓ 
 *▓▓ *▓▓▓  ▓▓ ▓▓▓▓▓▓▓▓ *▓▓ ▓▓*  ▓▓        ▓▓▓ ▓****** ▓▓▓   *▓▓ ▓▓▓  ▓▓ *▓* ▓▓   ▓▓▓▓  
 *▓▓  ▓▓▓  ▓▓ ▓▓*** ▓▓**▓▓ *▓▓  ▓▓▓▓**▓▓▓▓▓▓*          ▓▓▓*▓▓▓▓ ▓▓▓  ▓▓ *▓* ▓▓   ▓▓▓*  
 ▓▓▓  ▓▓▓  ▓▓*▓▓    ▓▓▓ ▓▓  ▓▓▓ ▓▓▓▓▓*▓▓▓▓▓▓            ▓▓▓▓▓▓  ▓▓▓  ▓▓ ▓▓* ▓▓▓   ▓▓   
  ▓    ▓   ** *      *  *    *  ****   *▓*               *▓*     ▓    *  *  ▓▓▓* ▓▓▓   
                                                                             *   ▓▓    
                                                                                 ▓▓    
                                                                                 *     
    """
    )

    # *****************************************3、获得模型参数和模型/log路径**********************************************
    model_parameters = parameters[train_parameters['train_algorithm']]
    # 创建相关的log和model的保存路径
    model_path = "./models/" + \
                 train_parameters['env_name'] + "/" + \
                 train_parameters['train_algorithm']
    log_path = "./logs/" + \
               train_parameters['env_name'] + "/" + \
               train_parameters['train_algorithm']
    if os.path.exists(model_path):
        pass
    else:
        os.makedirs(model_path)
    if os.path.exists(log_path):
        pass
    else:
        os.makedirs(log_path)

    # *****************************************4、打印训练参数、模型参数、相关路径*****************************************
    # （1）打印训练参数
    print("++++++++++++++++++++++++训练参数+++++++++++++++++++++++++++")
    for key in train_parameters.keys():
        print("\t{0:^20}\t{1:^20}".format(key + ":", str(train_parameters[key])))
    print("\n")
    # （2）打印模型参数
    print("++++++++++++++++++++++++模型参数+++++++++++++++++++++++++++")
    for key in model_parameters.keys():
        print("\t{0:^20}\t{1:^20}".format(key + ":", str(model_parameters[key])))
    print("\n")
    # （3）打印相关路径
    print("++++++++++++++++++++++++相关路径+++++++++++++++++++++++++++")
    print("\t" + "模型保存路径" + ":" + "\t" + str(model_path))
    print("\t" + "log保存路径" + ":" + "\t" + str(log_path))
    print("\n")

    # *****************************************5、开始训练或者推断******************************************************
    if train_parameters["train"]:
        train(train_parameters, model_parameters, model_path, log_path)
    elif train_parameters["inference"]:
        inference(train_parameters, model_parameters, model_path)

# python train.py --env-name=USV-AG-Scenario2-4v4 --env-run-type=exe --train-algorithm=PPO_DISCRETE --group-num=1 --agent-group-index 0 0 0 0  --max-episode-len=500 --num-episodes=5000 --train-frequency=50 --print-frequency=1 --inference