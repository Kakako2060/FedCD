import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # 进度条

def process_data(config, mode="train"):
    """
    处理训练数据或测试数据，并保存为 .pt 文件。

    参数：
        config: 配置字典，包含数据路径和参数
        mode: "train" 或 "test"，用于指定当前处理的是训练数据还是测试数据
    """
    root_dir = config['root_dir']
    data_dir = os.path.join(root_dir, mode)  # train 或 test 目录
    img_size = config.get('img_size', 224)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std= (0.229, 0.224, 0.225))
    ])

    # 读取公用 labels.csv 文件
    labels_df = pd.read_csv(config['labels_path'], header=None, names=['img_name', 'label'])
    label_dict = dict(zip(labels_df['img_name'], labels_df['label']))

    clients_dir = config[f'{mode}_clients_dir']
    os.makedirs(config['output_dir'], exist_ok=True)

    client_files = sorted([f for f in os.listdir(clients_dir) if f.endswith('.csv')])

    all_user_data = {}
    all_users = []
    num_samples = []

    print(f"\n开始处理 {mode} 数据，共 {len(client_files)} 个客户端...\n")

    for idx, client_csv in enumerate(tqdm(client_files, desc=f"处理中 ({mode})", unit="client")):
        client_id = f"{idx:05d}"
        user_id = f"f_{client_id}"

        client_df = pd.read_csv(os.path.join(clients_dir, client_csv), header=None, names=['img_name'])

        images, labels = [], []

        for img_name in tqdm(client_df['img_name'], desc=f"处理 {user_id}", leave=False, unit="img"):
            img_path = os.path.join(data_dir, img_name)

            if not os.path.exists(img_path) or img_name not in label_dict:
                continue  # 跳过缺失或无标签数据

            try:
                img = Image.open(img_path).convert('RGB')
                images.append(transform(img))
                labels.append([label_dict[img_name]])  # Nx1 形状
            except Exception as e:
                print(f"处理 {img_path} 失败: {e}")
                continue

        if len(images) == 0:
            print(f"客户端 {user_id} 无有效数据，跳过")
            continue

        all_users.append(user_id)
        all_user_data[user_id] = {'x': torch.stack(images), 'y': torch.tensor(labels).int()}
        num_samples.append(len(images))

    dataset = {'users': all_users, 'user_data': all_user_data, 'num_samples': num_samples}
    torch.save(dataset, os.path.join(config['output_dir'], f"{mode}.pt"))

    print(f"\n{mode.capitalize()} 数据已保存。\n")


def split_test_data(config):
    """
    读取 test.csv 并随机划分为 12 份，每个客户端分配一部分测试数据
    """
    root_dir = config['root_dir']
    test_dir = os.path.join(root_dir, 'test')
    num_clients = 12  # 客户端数量

    # 读取 test.csv
    test_df = pd.read_csv(config['test_csv_path'], header=None, names=['img_name'])

    # 随机打乱数据并划分
    shuffled_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_indices = np.array_split(range(len(shuffled_df)), num_clients)

    os.makedirs(config['test_clients_dir'], exist_ok=True)

    for i, indices in enumerate(split_indices):
        client_df = shuffled_df.iloc[indices]
        client_df.to_csv(os.path.join(config['test_clients_dir'], f"{i:05d}.csv"), header=False, index=False)

    print("\n测试数据已随机划分为 12 份。\n")


if __name__ == "__main__":
    config = {
        'root_dir': '.',
        'train_clients_dir': './12_clients/split_real/',
        'test_clients_dir': './12_clients_test/split_real/',
        'labels_path': './labels.csv',  # 训练和测试共用 labels.csv
        'test_csv_path': './test.csv',  # 用于划分测试数据
        'output_dir': './processed_data',
        'img_size': 224
    }

    # 随机划分测试数据
    split_test_data(config)

    # 处理训练数据
    # process_data(config, mode="train")

    # 处理测试数据（已划分到 12 个客户端）
    # process_data(config, mode="test")
