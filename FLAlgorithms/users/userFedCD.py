
from sklearn.cluster import DBSCAN
from FLAlgorithms.users.userbase import User
from FLAlgorithms.trainmodel.gan_models import *
from utils.model_utils import *
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score
from utils.model_utils import  METRICS

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
import copy
MIN_SAMPLES_PER_LABEL = 1


class UserFedCD(User):
    def __init__(self,
                 args, id, model,
                 train_data, test_data,
                 available_labels, label_info,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.num_glob = args.num_glob_iters
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.id = id
        self.metrics = {key: [] for key in METRICS}
        self.dataset_name = get_dataset_name(self.dataset)
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.available_labels = available_labels
        self.label_info = label_info
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.nll_loss = nn.NLLLoss().to(device)
        self.prev_model = None
        self.base_seed = 2025
        self.g = torch.Generator()
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True,
                                      generator=self.g, num_workers=0
                                      )
        labels = []
        with torch.no_grad():
            for _, y in DataLoader(train_data, batch_size=32, shuffle=False):
                labels.extend(y.tolist())
        label_counts = Counter(labels)
        total_samples = len(labels)
        avg_count = total_samples / len(label_counts) if label_counts else 0
        core_labels = [lb for lb, cnt in label_counts.items() if cnt > avg_count]
        core_samples = sum(label_counts[lb] for lb in core_labels)
        self.core_ratio = core_samples / total_samples if total_samples else 0.0
        max_count = max(label_counts.values())
        self.core_ratio = max_count / len(labels)
        print(f"gini = {self.core_ratio:.3f},{label_counts}")
        self.lambda_min = args.lambda_min
        self.lambda_max = args.lambda_max
        self.training_ratio = args.ratio
        self.min_sample =args.min_sample
        self.min_samples_factor = 800
        self.w=0.1

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

    def train(self, user_id, glob_iter, early_stop=10, count_labels=True):
        self.prev_model = copy.deepcopy(self.model)
        self.clean_up_counts()
        self.model.train()
        train_loss, num_batches, alignment_loss = 0, 0, 0
        for epoch in range(1, self.local_epochs + 1):
            epoch_seed = self.base_seed - glob_iter - epoch
            self.g.manual_seed(epoch_seed)

            if glob_iter < early_stop:
                # Warm start阶段：正常训练
                for i, (X, y) in enumerate(self.trainloader):
                    X, y = X.to(device), y.to(device)
                    self.optimizer.zero_grad()
                    user_output = self.model(X)
                    loss = self.loss(user_output['output'], y)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1
            else:
                # 聚类感知训练阶段
                num_batches_total = len(self.trainloader)
                if num_batches_total < 10:
                    # 小批量直接训练
                    train_mask = np.ones(num_batches_total, dtype=bool)
                    lambdav = self.lambda_min
                else:
                    # 执行聚类分析
                    gradients = self._compute_gradients(self.trainloader)
                    is_outlier, sil, ratio_geom, min_samples, eps = self._perform_clustering(
                        gradients, num_batches_total, user_id, glob_iter, epoch)

                    # 确定训练策略和lambda权重
                    lambdav = self._calculate_lambda_weight(glob_iter, early_stop)

                    if ((glob_iter - early_stop) % self.num_glob) < int(self.num_glob * self.training_ratio):
                        train_mask = ~is_outlier
                    else:
                        train_mask = is_outlier

                    print(f"[U{user_id:02d}-R{glob_iter:03d}-E{epoch}] "
                          f"ratio_geom={np.clip(ratio_geom, 0.0, 1.0):.3f} λ={lambdav:.3f} "
                          f"r={self.training_ratio:.2f} sil={sil:.3f} "
                          f"core%={1 - is_outlier.mean():.2f} min_samples={min_samples} eps={eps:.3f}")
                # 基于mask的训练
                E = 0
                for i, (X, y) in enumerate(self.trainloader):
                    if not train_mask[i]:
                        E += 1
                        continue
                    X, y = X.to(device), y.to(device)
                    self.optimizer.zero_grad()
                    user_output = self.model(X)
                    current_features = user_output['features']
                    loss = self.loss(user_output['output'], y)

                    with torch.no_grad():
                        prev_features = self.prev_model(X)['features']
                    feature_alignment_loss = self._compute_feature_alignment_loss(
                        current_features, prev_features.detach(), loss)
                    total_loss = loss + lambdav * feature_alignment_loss
                    total_loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    alignment_loss += feature_alignment_loss.item()
                    num_batches += 1

        self.lr_scheduler.step()
        # 计算平均损失
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        return avg_train_loss

    def _compute_gradients(self, dataloader):
        """计算每个batch的梯度方向"""
        gradients = []
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = self.model(X)['output']
            self.optimizer.zero_grad()
            loss = self.loss(outputs, y)
            grads = torch.autograd.grad(
                loss, list(self.model.classifier.parameters()), retain_graph=False)
            grad_direction = torch.cat([g.flatten() for g in grads]).detach()
            grad_norm = torch.norm(grad_direction).unsqueeze(0)
            grad_direction = grad_direction / torch.norm(grad_direction)
            grad_direction = torch.cat([grad_norm, grad_direction])
            gradients.append(grad_direction)
            del X, y, outputs, loss, grads, grad_direction, grad_norm
        return torch.stack(gradients)

    def _perform_clustering(self, gradients, num_batches, user_id, glob_iter, epoch):
        """执行梯度聚类并返回训练mask"""
        # 归一化梯度范数
        gn_col = gradients[:, 0]
        gn_min = gn_col.min()
        gn_max = gn_col.max()
        gn_col = (gn_col - gn_min) / (gn_max - gn_min + 1e-8)
        gradients[:, 0] = gn_col

        # PCA降维
        X = gradients - gradients.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        cum_var = (S ** 2).cumsum(0) / (S ** 2).sum()
        k = int((cum_var >= 0.99).nonzero()[0]) + 1
        low_dim = X @ Vh[:k].T

        # 计算距离矩阵
        euc_dist = torch.cdist(low_dim, low_dim, p=2).detach()

        # 确定min_samples参数
        min_samples = min(
            self.min_sample + int(num_batches * self.batch_size / self.min_samples_factor),
            num_batches
        )

        # 使用肘部法则确定eps
        k2 = min_samples - 1
        k_dist = torch.kthvalue(euc_dist, k2, dim=1)[0].detach()
        sorted_dist = torch.sort(k_dist)[0].detach()
        second_diff2 = torch.diff(sorted_dist, n=2)
        elbow_idx_diff2 = torch.argmax(second_diff2) + 2

        ratio_geom = (self.core_ratio * self.w) + ((elbow_idx_diff2 / num_batches) * (1-self.w))
        elbow_idx = max(int(num_batches * 0.2),
                        min(int(num_batches * ratio_geom), int(num_batches * 0.8)))
        eps = sorted_dist[elbow_idx].item()
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(
            euc_dist.cpu().numpy())
        is_outlier = (labels == -1)
        low_dim_np = low_dim.cpu().numpy().astype(np.float64)
        sil = silhouette_score(low_dim_np, labels) if len(set(labels)) > 1 else -1

        # 可视化（仅在特定条件下）
        if epoch == 1 and sil > 0.5 and glob_iter % 5 == 0:
            self._visualize_clusters(low_dim_np, labels, user_id, glob_iter, epoch, sil)

        del low_dim, k_dist, sorted_dist, low_dim_np
        torch.cuda.empty_cache()

        return is_outlier, sil, ratio_geom, min_samples, eps

    def _visualize_clusters(self, low_dim_data, labels, user_id, glob_iter, epoch, sil_score):
        try:
            tsne = TSNE(n_components=2, random_state=42,
                        perplexity=min(30, low_dim_data.shape[0] - 1))
            tsne_results = tsne.fit_transform(low_dim_data)

            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'mathtext.fontset': 'stix',
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })

            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels = set(labels) - {-1}
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

            # 绘制非边缘批次
            normal_indices = np.where(labels != -1)[0]
            if len(normal_indices) > 0:
                for label in unique_labels:
                    cluster_indices = normal_indices[labels[normal_indices] == label]
                    if len(cluster_indices) > 0:
                        ax.scatter(tsne_results[cluster_indices, 0],
                                   tsne_results[cluster_indices, 1],
                                   c=[color_map[label]],
                                   s=60, alpha=0.8, edgecolors='white', linewidth=0.5,
                                   label=f'Cluster {label}', marker='o')

            # 绘制边缘批次
            outlier_indices = np.where(labels == -1)[0]
            if len(outlier_indices) > 0:
                ax.scatter(tsne_results[outlier_indices, 0],
                           tsne_results[outlier_indices, 1],
                           c='red', marker='X', s=80, alpha=0.9,
                           edgecolors='darkred', linewidth=0.8, label='Outliers')

            ax.set_xlabel('t-SNE Dimension 1', fontsize=14, labelpad=10)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=14, labelpad=10)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True)
            ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)

            os.makedirs('cluster_visualizations', exist_ok=True)
            filename = f'cluster_visualizations/user_{user_id}_glob_{glob_iter}_epoch_{epoch}_{self.dataset}_{sil_score:.3f}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved: {filename}")
        except Exception as e:
            print(f"t-SNE visualization failed: {e}")

    def _compute_feature_alignment_loss(self, current_features, prev_features, base_loss):
        """计算特征对齐损失"""
        if current_features.dim() == 4:
            B, C, H, W = current_features.shape
            current_features = current_features.view(B, -1)
            prev_features = prev_features.view(B, -1)

        ln = nn.LayerNorm(current_features.size(1), elementwise_affine=False).to(current_features.device)
        current_norm = ln(current_features)
        prev_norm = ln(prev_features)
        mse_loss = F.mse_loss(current_norm, prev_norm)
        mse_loss = torch.clamp(mse_loss, max=10.0 * base_loss)
        return mse_loss

    def _calculate_lambda_weight(self, glob_iter, early_stop):
        """计算动态lambda权重"""
        stage_ratio = max(0.0, (glob_iter - early_stop) / (self.num_glob - early_stop + 1e-8))

        if ((glob_iter - early_stop) % self.num_glob) < int(self.num_glob * self.training_ratio):

            return self.lambda_min
        else:

            return self.lambda_min + (self.lambda_max - self.lambda_min) * stage_ratio
