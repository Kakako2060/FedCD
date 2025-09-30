import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
from utils.model_utils import get_log_path, METRICS
import seaborn as sns
import string
import matplotlib.colors as mcolors
import os
COLORS=list(mcolors.TABLEAU_COLORS)
MARKERS=["o", "v", "s", "*", "x", "P"]

plt.rcParams.update({'font.size': 14})
n_seeds=3

def load_results(args, algorithm, seed):
    alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
    hf = h5py.File("./{}/{}.h5".format(args.result_path, alg), 'r')
    metrics = {}
    for key in METRICS:
        metrics[key] = np.array(hf.get(key)[:])
    return metrics


def get_label_name(name):
    name = name.split("_")[0]
    if 'Distill' in name:
        if '-FL' in name:
            name = 'FedDistill' + r'$^+$'
        else:
            name = 'FedDistill'
    elif 'FedDF' in name:
        name = 'FedFusion'
    elif 'FedEnsemble' in name:
        name = 'Ensemble'
    elif 'FedAvg' in name:
        name = 'FedAvg'
    elif 'FedGen' in name:
        name = 'FedGen'
    elif 'FedProx' in name:
        name = 'FedProx'
    elif 'FedAPEN' in name:
        name = 'FedAPEN'
    elif 'FedMut' in name:
        name = 'FedMut'
    elif 'FedAWARE' in name:
        name = 'FedAWARE'
    elif 'FedFOR' in name:
        name = 'FedCD'
    return name

def plot_train_loss(args, algorithms):
    """绘制训练损失曲线"""
    n_seeds = args.times
    dataset_ = process_dataset_name(args.dataset)

    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(algorithms):
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        # 提取训练损失数据
        all_loss = np.concatenate([metrics[seed]['train_loss'] for seed in range(n_seeds)])

        # 计算统计量
        length = len(all_loss) // n_seeds
        mean_loss = np.mean(all_loss.reshape(n_seeds, length), axis=0)

        # 绘制曲线
        sns.lineplot(
            x=np.arange(length) + 1,
            y=mean_loss,
            color=COLORS[i],
            label=get_label_name(algorithm),
            linewidth=2.5,
            errorbar=None,
            marker='',
            markersize=8,
            markevery=int(length/10)
        )

    # 图表装饰
    plt.grid(linestyle='--', alpha=0.6)

    plt.xlabel("Communication rounds")
    plt.ylabel("Training Loss ")

    # 动态调整Y轴范围
    ymin, ymax = np.percentile(all_loss, [5, 95])
    plt.ylim(ymin*0.9, ymax*1.1)

    # 保存路径
    save_path = os.path.join('figs', f"{dataset_[0]}_train_loss.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Training loss plot saved to {save_path}")

def process_dataset_name(dataset_str):
    """统一处理数据集名称"""
    dataset_ = dataset_str.split('-')
    dataset_[0] = dataset_[0].capitalize()

    # 名称校正
    name_map = {
        "Cifar": "Cifar10",
        "Cifa100": "Cifar100",
        "Mnist": "MNIST",
        "Fmnist": "FashionMNIST"
    }
    dataset_[0] = name_map.get(dataset_[0], dataset_[0])

    return dataset_


def plot_loss_comparison(args, algorithms):
    """绘制训练-测试损失对比图（单Y轴，截断FedGen高值）"""
    n_seeds = args.times
    dataset_ = process_dataset_name(args.dataset)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 阶段1：先绘制其他算法确定坐标范围
    max_val = 0  # 记录常规算法最大值
    regular_algs = [alg for alg in algorithms if alg != "FedGen"]

    for algorithm in regular_algs:
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        train_loss = np.concatenate([m['train_loss'] for m in metrics])
        test_loss = np.concatenate([m['glob_loss'] for m in metrics])
        loss_diff = np.abs(train_loss - test_loss).reshape(n_seeds, -1).mean(0)

        # 更新最大常规值
        current_max = loss_diff.max()
        if current_max > max_val:
            max_val = current_max

        sns.lineplot(
            x=np.arange(len(loss_diff)) + 1,
            y=loss_diff,
            label=get_label_name(algorithm),
            linewidth=1.2,
            alpha=0.8
        )

    # 阶段2：固定Y轴范围后绘制FedGen
    if "FedGen" in algorithms:
        metrics = [load_results(args, "FedGen", seed) for seed in range(n_seeds)]
        train_loss = np.concatenate([m['train_loss'] for m in metrics])
        test_loss = np.concatenate([m['glob_loss'] for m in metrics])
        loss_diff = np.abs(train_loss - test_loss).reshape(n_seeds, -1).mean(0)

        # 设置Y轴上限（常规算法最大值的1.1倍）
        plt.ylim(0, max_val * 1.1)

        # 绘制FedGen（超出部分自动截断）
        sns.lineplot(
            x=np.arange(len(loss_diff)) + 1,
            y=loss_diff,
            label=get_label_name("FedGen") ,
            linewidth=1.2,
            alpha=0.8,
            color='purple'  # 用特殊颜色标注
        )

    # 统一标注
    plt.ylabel("|Train Loss - Test Loss|")
    plt.xlabel("Communication rounds")
    plt.grid(
        which='both',
        linestyle=':',
        linewidth=0.8,
        alpha=0.4,
        color='slategray'
    )
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # 底部居中
        ncol=4,  # 分4列显示
        fontsize=11,
        framealpha=0.9,
        columnspacing=1.5
    )

    # 保存
    save_path = os.path.join('figs', f"{dataset_[0]}_loss_comparison.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Loss comparison plot saved to {save_path}")


def plot_results(args, algorithms):
    n_seeds = args.times
    dataset_ = args.dataset.split('-')
    dataset_[0] = dataset_[0].capitalize()  # 统一首字母大写
    if dataset_[0] == "Cifar":
        dataset_[0] = "Cifar10"            # 修正Cifar->Cifar10
    elif dataset_[0] == "Cifa100":
        dataset_[0] = "Cifar100"
    sub_dir = dataset_[0] + "/" + dataset_[2] # e.g. Mnist/ratio0.5
    # os.system("mkdir -p figs/{}".format(sub_dir))  # e.g. figs/Mnist/ratio0.5
    fig_dir = os.path.join('figs', sub_dir)
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(1, figsize=(10, 6))
    TOP_N = 2
    max_acc = 0
    for i, algorithm in enumerate(algorithms):
        algo_name = get_label_name(algorithm)
        ######### plot test accuracy ############
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        all_curves = np.concatenate([metrics[seed]['glob_acc'] for seed in range(n_seeds)])
        top_accs =  np.concatenate([np.sort(metrics[seed]['glob_acc'])[-TOP_N:] for seed in range(n_seeds)] )
        acc_avg = np.mean(top_accs)
        acc_std = np.std(top_accs)
        info = 'Algorithm: {:<10s}, Accuracy = {:.2f} %, deviation = {:.2f}'.format(algo_name, acc_avg * 100, acc_std * 100)
        print(info)
        length = len(all_curves) // n_seeds
        sns.lineplot(
            x=np.array(list(range(length)) * n_seeds) + 1,
            y=all_curves.astype(float),
            legend='brief',
            color=COLORS[i],
            label=algo_name,
            errorbar=None,
            linewidth=1.2,                  # 加粗主曲线
            alpha=0.9,                      # 适度透明
            markersize=8,                   # 标记尺寸
            markeredgewidth=1.5,            # 标记边缘粗细
            markevery=int(length/10),       # 每10个点显示一个标记

        )

    plt.gcf()
    plt.grid()
    plt.ylabel("Test Accuracy (%)", fontsize=14)  # 添加Y轴标签
    plt.xticks(fontsize=12)                 # X轴刻度字号
    plt.yticks(fontsize=12)                 # Y轴刻度字号
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.0%}'))  # 百分比格式
    plt.legend(
        loc='lower right',            # 定位到右下角
        frameon=True,                 # 添加背景框
        fontsize=12,                  # 字号统一
        title_fontsize='13',          # 图例标题字号
        edgecolor='black',            # 边框颜色
        ncol=1                       # 分两列显示（避免过长）
    )
    plt.xlabel('Communication rounds')
    # max_acc = np.max([max_acc, np.max(all_curves) ]) + 4e-2
    #
    # if args.min_acc < 0:
    #     alpha = 0.7
    #     min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1-alpha)
    # else:
    #     min_acc = args.min_acc
    # plt.ylim(min_acc, max_acc)
    dynamic_padding = (np.max(all_curves) - np.min(all_curves)) * 0.1  # 动态留白
    max_acc = np.max(all_curves) + dynamic_padding
    min_acc = np.min(all_curves) - dynamic_padding*0.5
    plt.ylim(min_acc, max_acc)
    fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1] + '-' + dataset_[2] + '.png')
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0, format='png', dpi=400)
    print('file saved to {}'.format(fig_save_path))
    plt.close()
    for i, algorithm in enumerate(algorithms):
        # 关键修复：重新获取算法名称
        algo_name = get_label_name(algorithm)  # 新增这行

        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        all_loss = np.concatenate([m['glob_loss'] for m in metrics])

        # 计算合理的数据长度
        loss_length = len(all_loss) // n_seeds  # 独立计算

        sns.lineplot(
            x=np.array(list(range(len(all_loss)//n_seeds)) * n_seeds) + 1,
            y=all_loss.astype(float),
            color=COLORS[i],
            label=algo_name,
            errorbar=None,
            linewidth=1.2,  # 将线宽从2.5减到1.2
            alpha=0.9,       # 提高透明度保持线条锐利
            marker='',   # 禁用数据点标记
            markersize=0,  # 确保不显示标记
            markevery=0    # 完全禁用间隔标记0T
        )

    # 优化坐标轴范围
    q95 = np.quantile(all_loss, 0.95)
    q5 = np.quantile(all_loss, 0.05)
    plt.ylim(max(q5 * 0.9, np.min(all_loss)), min(q95 * 1.1, np.max(all_loss)))

    # 增强图例可读性
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # 底部居中
        ncol=4,  # 分4列显示
        fontsize=11,
        framealpha=0.9,
        columnspacing=1.5
    )
    # 增强网格样式
    plt.grid(
        which='both',
        linestyle=':',
        linewidth=0.8,
        alpha=0.4,
        color='slategray'
    )

    # 优化标题和标签
    plt.xlabel("Communication Rounds", fontsize=14)
    plt.ylabel("Test Loss ", fontsize=14)
    plt.xticks(np.arange(0, 201, 20), fontsize=10)  # 调整刻度间隔和字体大小



    # 保存测试损失图
    loss_fig_path = os.path.join('figs', sub_dir, f"{dataset_[0]}-{dataset_[1]}-test_loss.png")
    # 提升保存质量
    plt.savefig(
        loss_fig_path,
        dpi=600,
        bbox_inches='tight',
        facecolor='white',  # 强制白底
        transparent=False
    )
    plt.close()  # 明确关闭图表
    print(f"Test loss plot saved to {loss_fig_path}")