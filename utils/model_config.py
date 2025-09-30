CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim
    'cifar': ([16, 'M', 32, 'M', 'F'], 3, 10, 128, 32),
    'cifa100': ([16, 'M', 32, 'M', 'F'], 3, 100, 128, 128),
    'cifar100-c25': ([32, 'M', 64, 'M', 128, 'F'], 3, 25, 128, 128),
    'cifar100-c30': ([32, 'M', 64, 'M', 128, 'F'], 3, 30, 2048, 128),
    'cifar100-c50': ([32, 'M', 64, 'M', 128, 'F'], 3, 50, 2048, 128),

    'emnist': ([6, 16, 'F'], 1, 26, 784, 32),
    'mnist': ([6, 16, 'F'], 1, 10, 784, 32),
    'mnist_cnn1': ([6, 'M', 16, 'M', 'F'], 1, 10, 64, 32),
    'mnist_cnn2': ([16, 'M', 32, 'M', 'F'], 1, 10, 128, 32),
    'celeb': ([16, 'M', 32, 'M', 64, 'M', 'F'], 3, 2, 64, 32)
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # input-dim, hidden_dimension, latent_dimension, input_channel, n_class, embedding_dim
    'cifar': ( 512, 32, 3, 10, 64),
    'cifa100': ( 512, 512, 3, 100, 64),
    'celeb': (972, 128, 32, 3, 2, 32),
    'mnist': (256, 32, 1, 10, 32),
    'mnist-cnn0': (256, 32, 1, 10, 64),
    'mnist-cnn1': ( 128, 32, 1, 10, 32),
    'mnist-cnn2': ( 64, 32, 1, 10, 32),
    'mnist-cnn3': ( 64, 32, 1, 10, 16),
    'emnist': ( 256, 32, 1, 26, 32),
    'emnist-cnn0': ( 256, 32, 1, 26, 64),
    'emnist-cnn1': ( 128, 32, 1, 26, 32),
    'emnist-cnn2': ( 128, 32, 1, 26, 16),
    'emnist-cnn3': ( 64, 32, 1, 26, 32),
}



RUNCONFIGS = {
    'emnist':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0, # adversarial student loss
            'unique_labels': 26,
            'generative_alpha':10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'mnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },

    'celeb':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 2,
            'generative_alpha': 10,
            'generative_beta': 10, 
            'weight_decay': 1e-2
        },
    'cifar':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 5,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 10,
            'generative_alpha': 10,
            'generative_beta': 10,
            'weight_decay': 1e-2
        },
    'cifa100':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 5,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 100,
            'generative_alpha': 10,
            'generative_beta': 10,
            'weight_decay': 1e-2
        },
    'retina':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 32,
            'ensemble_epochs': 5,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 2,
            'generative_alpha': 10,
            'generative_beta': 10,
            'weight_decay': 1e-4
        },
    'covid':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 32,
            'ensemble_epochs': 5,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 3,
            'generative_alpha': 10,
            'generative_beta': 10,
            'weight_decay': 1e-4
        },
    'sentiment140': {  # [!code ++]
        'ensemble_lr': 3e-5,            # 更小的学习率适配文本任务
        'ensemble_batch_size': 32,      # 减小批次大小防止显存溢出
        'ensemble_epochs': 10,          # 文本训练通常需要较少epoch
        'num_pretrain_iters': 5,        # 预训练迭代次数
        'ensemble_alpha': 1,            # 教师模型损失权重
        'ensemble_beta': 0.1,           # 对抗损失权重（根据论文调整）
        'unique_labels': 2,             # 二分类任务
        'generative_alpha': 5,          # 生成模型调节参数
        'generative_beta': 2,           # 对抗生成权重
        'weight_decay': 1e-3            # 更强的正则化防止过拟合
    }
}

