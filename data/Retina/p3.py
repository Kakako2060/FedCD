import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def process_and_save(config, mode="train"):
    root_dir = config['root_dir']
    data_dir = Path(os.path.join(root_dir, mode))  # train 或 test 目录
    img_size = config.get('img_size', 256)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4915, 0.4823, 0.4468),
                             std=(0.2470, 0.2435, 0.2616))
    ])

    labels_path = config[f'{mode}_labels_path']
    labels_df = pd.read_csv(labels_path, header=None, names=['img_name', 'label'])
    label_dict = dict(zip(labels_df['img_name'], labels_df['label']))

    clients_dir = config[f'{mode}_clients_dir']
    os.makedirs(config['output_dir'], exist_ok=True)

    client_files = sorted([f for f in os.listdir(clients_dir) if f.endswith('.csv')])

    all_user_data = {}
    all_users = []
    num_samples = []
    client_category_counts = {}

    print(f"\n开始处理 {mode} 数据，共 {len(client_files)} 个客户端...\n")

    for idx, client_csv in enumerate(client_files):
        client_id = f"{idx:05d}"
        user_id = f"f_{client_id}"
        client_category_counts[user_id] = {category: 0 for category in set(labels_df['label'])}

        client_df = pd.read_csv(os.path.join(clients_dir, client_csv), header=None, names=['img_name'])

        images = []
        labels = []

        for path in tqdm(list(data_dir.glob("*.npy"))):
            name = path.name
            if name in client_df['img_name'].to_list():
                img = np.load(path).astype(np.uint8)
                tensor_img = transform(img)#torch.from_numpy(img).float()
                images.append(tensor_img)
                labels.append(label_dict[name])
                client_category_counts[user_id][label_dict[name]] += 1  # 更新类别计数

        if len(images) == 0:
            print(f"客户端 {user_id} 无有效数据，跳过")
            continue

        image_tensors = torch.stack(images)
        label_tensors = torch.tensor(labels).long()

        all_users.append(user_id)
        all_user_data[user_id] = {'x': image_tensors, 'y': label_tensors}
        num_samples.append(len(images))

    dataset = {
        'users': all_users,
        'user_data': all_user_data,
        'num_samples': num_samples
    }
    save_path = os.path.join(config['output_dir'], f"{mode}.pt")
    torch.save(dataset, save_path)

    print(f"\n所有 {mode} 数据已保存到 {save_path} 文件。\n")
    return client_category_counts

def plot_stacked_bar_chart(client_category_counts, categories, mode="train"):
    fig, ax = plt.subplots(figsize=(10, 6))
    clients = list(client_category_counts.keys())
    y_pos = range(len(clients))

    for i, category in enumerate(categories):
        values = [client_category_counts[client][category] for client in clients]
        ax.barh(y_pos, values, left=[sum(values[:i]) for values in zip(*values)], label=category)

    ax.set_xlabel('Number of Samples')
    ax.set_title(f'Sample Counts by Category ({mode})')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clients)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    config = {
        'root_dir': '.',
        'train_clients_dir': './5_clients/split_1',
        'test_clients_dir': './5_clients_test/split_1',
        'train_labels_path': './labels.csv',
        'test_labels_path': './labels.csv',
        'output_dir': './processed_data',
        'img_size': 256
    }

    train_counts = process_and_save(config, mode="train")
    test_counts = process_and_save(config, mode="test")

    # categories = list(train_counts[next(iter(train_counts))].keys())
    # plot_stacked_bar_chart(train_counts, categories, mode="train")
    # plot_stacked_bar_chart(test_counts, categories, mode="test")