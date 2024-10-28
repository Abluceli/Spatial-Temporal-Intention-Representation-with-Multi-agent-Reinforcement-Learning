parameters = {
    "DDPG": {
        "buffer_size": 51200,
        "lr_actor": 1.0e-3,
        "lr_critic": 1.0e-3,
        "sigma": 0.15,
        "gamma": 0.99,
        "batch_size": 512,
        "max_replay_buffer_len": 10240,
        "tau": 0.01
    },
    "DGN": {
        "buffer_size": 100000,
        "lr": 1.0e-3,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "gamma": 0.95,
        "batch_size": 1024,
        "max_replay_buffer_len": 10240,
        "tau": 1
    },
    "QMIX": {
        "buffer_size": 100000,
        "lr": 1.0e-2,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.1,
        "gamma": 0.95,
        "batch_size": 1024,
        "max_replay_buffer_len": 10240,
        "tau": 1,
        "grad_norm_clip": 10
    },
    "DQN": {
        "buffer_size": 100000,
        "lr": 1.0e-4,
        "epsilon": 1,
        "epsilon_decay": 10000,
        "epsilon_min": 0.05,
        "gamma": 0.99,
        "batch_size": 512,
        "max_replay_buffer_len": 10240,
        "tau": 1,
        "grad_norm_clip": 10
    },
    "INTENTION_DQN": {
        "buffer_size": 50000,
        "lr": 1.0e-4,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.1,
        "gamma": 0.99,
        "batch_size": 1024,
        "max_replay_buffer_len": 1024,
        "tau": 1,
        "grad_norm_clip": 10
    },
    "VDN": {
        "buffer_size": 100000,
        "lr": 1.0e-2,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.1,
        "gamma": 0.95,
        "batch_size": 1024,
        "max_replay_buffer_len": 10240,
        "tau": 1,
        "grad_norm_clip": 10
    },
    "MADDPG": {
        "buffer_size": 1000000,
        "lr_actor": 1.0e-3,
        "lr_critic": 1.0e-3,
        "sigma": 0.15,
        "gamma": 0.99,
        "batch_size": 1024,
        "max_replay_buffer_len": 10240,
        "tau": 0.01
    },
    "H2G_MAAC": {
        "buffer_size": 1000000,
        "lr_actor": 1.0e-3,
        "lr_critic": 1.0e-3,
        "sigma": 0.15,
        "gamma": 0.99,
        "batch_size": 128,
        "max_replay_buffer_len": 256,
        "tau": 0.01
    },
    "SAC": {
        "buffer_size": 100000,
        "soft_q_lr": 5.0e-4,
        "policy_lr": 5.0e-4,
        "alpha_lr": 5.0e-4,
        "gamma": 0.99,
        "alpha": 0.2,
        "batch_size": 1024,
        "max_replay_buffer_len": 10240,
        "tau": 0.01,
        "dynamic_alpha": True
    },
    "SAC_DISCRETE": {
        "buffer_size": 20480,
        "soft_q_lr": 5.0e-4,
        "policy_lr": 5.0e-4,
        "alpha_lr": 5.0e-4,
        "gamma": 0.99,
        "alpha": 0.2,
        "batch_size": 512,
        "max_replay_buffer_len": 2048,
        "tau": 0.01,
        "dynamic_alpha": True
    },
    "PPO": {
        "UPDATE_STEPS": 10,  # 10
        "epsilon": 0.2,
        "gamma": 0.99,
        "A_LR": 0.0003,  # 0.00001
        "C_LR": 0.0003,  # 0.00002
        'lambda': 0.95,  # GAE parameter
        'use_gae_adv': True,
        "entropy_dis": 0.005
    },
    "PPO_DISCRETE": {
        "batch_size": 512,
        "buffer_size": 10240,
        "num_epoch": 3,  # 10
        "epsilon": 0.2,
        "gamma": 0.99,
        "learning_rate": 0.0003,  # 0
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        'lambda': 0.95,  # GAE parameter
        'use_gae_adv': True,
        'beta': 0.005
    }
}
