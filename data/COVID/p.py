import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # 引入 tqdm 用于显示进度条

def process_and_save(config, mode="train"):
    """
    处理训练集或测试集数据，并保存为 .pt 文件

    参数：
        config: 配置字典，包含数据路径和参数
        mode: "train" 或 "test"，用于指定当前处理的是训练数据还是测试数据
    """
    # 初始化配置
    root_dir = config['root_dir']
    data_dir = os.path.join(root_dir, mode)  # train 或 test 目录
    img_size = config.get('img_size', 224)

    # 定义预处理管道
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std= (0.229, 0.224, 0.225))
    ])

    # 读取标签文件
    labels_path = config[f'{mode}_labels_path']
    labels_df = pd.read_csv(labels_path, header=None, names=['img_name', 'label'])
    label_dict = dict(zip(labels_df['img_name'], labels_df['label']))

    # 读取客户端划分
    clients_dir = config[f'{mode}_clients_dir']
    os.makedirs(config['output_dir'], exist_ok=True)

    client_files = sorted([f for f in os.listdir(clients_dir) if f.endswith('.csv')])

    all_user_data = {}
    all_users = []
    num_samples = []

    print(f"\n开始处理 {mode} 数据，共 {len(client_files)} 个客户端...\n")

    for idx, client_csv in enumerate(client_files):
        client_id = f"{idx:05d}"  # 确保五位编号
        user_id = f"f_{client_id}"

        # 读取客户端数据
        client_df = pd.read_csv(os.path.join(clients_dir, client_csv), header=None, names=['img_name'])

        # 初始化存储
        images = []
        labels = []

        for img_name in tqdm(client_df['img_name']):
            img_path = os.path.join(data_dir, img_name)

            # 跳过缺失文件
            if not os.path.exists(img_path):
                print(f"警告：跳过缺失文件 {img_path}")
                continue

            # 跳过无标签数据
            if img_name not in label_dict:
                print(f"警告：跳过无标签图片 {img_name}")
                continue

            # 处理图片
            try:
                img = Image.open(img_path).convert('RGB')
                tensor_img = transform(img)  # 应用预处理
                images.append(tensor_img)
                labels.append(label_dict[img_name])  # 转换为 Nx1
            except Exception as e:
                print(f"处理 {img_path} 失败: {str(e)}")
                continue

        # 转换为张量
        if len(images) == 0:
            print(f"客户端 {user_id} 无有效数据，跳过")
            continue

        # 堆叠图像和标签张量
        image_tensors = torch.stack(images)
        label_tensors = torch.tensor(labels).long()  # Nx1 形状

        # 存储数据
        all_users.append(user_id)
        all_user_data[user_id] = {'x': image_tensors, 'y': label_tensors}
        num_samples.append(len(images))

    # 保存最终数据
    dataset = {
        'users': all_users,
        'user_data': all_user_data,
        'num_samples': num_samples
    }
    save_path = os.path.join(config['output_dir'], f"{mode}.pt")
    torch.save(dataset, save_path)

    print(f"\n所有 {mode} 数据已保存到 {save_path} 文件。\n")

if __name__ == "__main__":
    config = {
        'root_dir': '.',
        'train_clients_dir': './12_clients/split_real',  # 训练集划分

        'test_clients_dir': './12_clients_test/split_real',  # 测试集划分
        'train_labels_path': './labels.csv',  # 训练标签
        'test_labels_path': './labels.csv',  # 测试标签
        'output_dir': './processed_data',
        'img_size': 224
    }

    # 处理训练数据
    process_and_save(config, mode="train")

    # 处理测试数据
    process_and_save(config, mode="test")
