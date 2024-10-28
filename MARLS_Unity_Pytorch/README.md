## MARLS（Multi-Agent Reinforcement Learning）
- **整合MAPE和Unity环境，实现多种多智能体强化学习算法**

<img src="C:\Users\liyang\Desktop\MARLS_Unity\get_logo_tool\logo.jpg" width="300" >

##目前版本
###版本：release 2.5
###新增功能：增加两个算法DGN和H2G-MAAC
- **DGN算法主要是基于DQN进行复现的，因此只支持离散的动作，且目前只在MAPE环境“simple_spread”,具体算法的命令行调用可参考下面的示例。**
- **H2G-MAAC算法主要是小论文中的方法，基于MADDPG进行实现，具体的命令行调用可参考下面的示例。**
## 介绍
```bash
2021/12/02  15:54    <DIR>    agents               # 强化学习算法复现
2021/11/30  15:06    <DIR>    buffers              # replay buffer复现
2021/12/30  15:06    <DIR>    data_analys_tools    # 数据分析工具包
2021/12/01  14:05    <DIR>    env_tools            # 环境工具包
2021/12/01  19:24    <DIR>    get_logo_tool        # 框架项目logo生成脚本
2021/11/29  14:06    <DIR>    logs                 # 训练结果保存
2021/12/01  14:12    <DIR>    mlagents_envs        # Unity环境脚本
2021/11/29  14:06    <DIR>    models               # 训练模型保存
2021/11/30  15:23             model_parameters.py  # 各个算法的超参数
2021/12/01  14:10    <DIR>    multiagent           # mape环境脚本
2021/12/02  16:25             README.md            # readme
2021/12/02  14:14             test.py              # 测试脚本
2021/12/02  16:10             train.py             # 训练主要脚本
2021/11/29  14:07    <DIR>    Unity_Envs           # Unity到处的可执行环境
```

## 环境配置
```bash
\MARLS_Unity>conda create -n MARLS_Unity python=3.7
\MARLS_Unity>conda activate MARLS_Unity
(MARLS_Unity) \MARLS_Unity>conda activate MARLS_Unity
(MARLS_Unity) \MARLS_Unity>pip install -r requirements.txt
```

##参数介绍
```bash
(MARLS_Unity) \MARLS_Unity>python train.py -h
usage: train.py [-h] [--env-type ENV_TYPE] [--env-name ENV_NAME]
                [--env-run-type ENV_RUN_TYPE] [--env-worker-id ENV_WORKER_ID]
                [--no-graphics] [--group-num GROUP_NUM]
                [--agent-group-index AGENT_GROUP_INDEX [AGENT_GROUP_INDEX ...]]
                [--share-parameters] [--train-algorithm TRAIN_ALGORITHM]
                [--train] [--inference] [--max-episode-len MAX_EPISODE_LEN]
                [--num-episodes NUM_EPISODES]
                [--save-frequency SAVE_FREQUENCY]
                [--train-frequency TRAIN_FREQUENCY]
                [--print-frequency PRINT_FREQUENCY] [--resume]
              
Multi-Agent Reinforcement learning
optional arguments:
  -h, --help             # show this help message and exit
  --env-type             # 训练环境的类型，"unity" 或者 "mape"
  --env-name             # unity或mape中环境的名称
  --env-run-type         # unity训练环境客户端训练还是可执行程序训练，"exe" or "client", 对于mape环境没有用
  --env-worker-id        # "exe"环境的worker_id, 可进行多环境同时训练, 默认0, 使用client是必须设为0!
  --no-graphics          # 使用unity训练时是否打开界面
  --group-num            # 环境中智能体 组/类别 的数量
  --agent-group-index    # 环境中每个agent对应的组编号
  --share-parameters     # 环境中每组智能体是否组间共享网络
  --train-algorithm      # 训练算法: 目前支持DDPG, MADDPG, SAC, PPO,且名称都需要大写
  --train                # 是否训练
  --inference            # 是否推断
  --max-episode-len      # 每个episode的最大step数
  --num-episodes         # episode数量
  --save-frequency       # 模型保存频率
  --train-frequency      # 模型训练频率, 1表示每个step调用训练函数一次
  --print-frequency      # 训练数据打印输出频率, 100表示每100轮打印一次
  --resume               # 是否按照上一次的训练结果，继续训练
  --discrete-action      # 环境中是否是离散的动作，这个目前主要是针对DGN的参数
  --graph-obs            # 环境中是否是图形式的观测，这个目前主要是针对H2G-MAAC算法的参数
```

## 怎么样训练和推断（包含断点继续训练）
###unity环境：
```bash
#unity环境训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=unity --env-name=Navigation --train-algorithm=SAC --no-graphics --group-num=1 --agent-group-index 0 0 0 --train

#unity环境继续训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=unity --env-name=Navigation --train-algorithm=SAC --no-graphics --group-num=1 --agent-group-index 0 0 0 --train --resume

#unity环境推断
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=unity --env-name=Navigation --train-algorithm=SAC --group-num=1 --agent-group-index 0 0 0 --inference
```
###mape环境：
```bash
#mape环境训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --train

#mape环境继续训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --train --resume

#mape环境推断
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --inference
```

###DGN算法命令行调用示例：
```bash
#mape环境训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_spread --train-algorithm=DGN --group-num=1 --agent-group-index 0 0 0 --discrete-action --max-episode-len=50 --train-frequency=50 --print-frequency=5 --train --no-graphics
```

###DGN算法命令行调用示例：
```bash
#mape环境训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=Predator-prey-4v4 --train-algorithm=H2G_MAAC --group-num=2 --agent-group-index 0 0 0 0 1 1 1 1 --graph-obs --max-episode-len=50 --train-frequency=50 --print-frequency=10 --train --no-graphics
```

