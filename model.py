import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.optim import Adam
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
from improved_tsne_visualizer import ImprovedTSNEVisualizer


class PromptModule(nn.Module):
    """Soft Prompt Construction Module"""

    def __init__(self, n_users, n_items, feat_dims, embed_dim, prompt_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.prompt_dim = prompt_dim

        # Feature reduction for each modality
        self.audio_reduce = nn.Linear(feat_dims['audio'], embed_dim)
        self.image_reduce = nn.Linear(feat_dims['image'], embed_dim)
        self.text_reduce = nn.Linear(feat_dims['text'], embed_dim)

        # Prompt transformation matrices
        self.W_p_audio = nn.Linear(embed_dim, prompt_dim)
        self.W_p_image = nn.Linear(embed_dim, prompt_dim)
        self.W_p_text = nn.Linear(embed_dim, prompt_dim)
        self.W_p_user = nn.Linear(embed_dim, prompt_dim)

    def forward(self, audio_feat, image_feat, text_feat, adj_mat, user_item_mat):
        """
            Construct soft prompts
            Returns: p_audio, p_image, p_text, p_user
            """
        # Reduce feature dimensions
        audio_emb = self.audio_reduce(audio_feat)
        image_emb = self.image_reduce(image_feat)
        text_emb = self.text_reduce(text_feat)

        # Normalize and create modality prompts (Eq. 4)
        p_audio = self.W_p_audio(F.normalize(audio_emb + audio_emb, dim=1))
        p_image = self.W_p_image(F.normalize(image_emb + image_emb, dim=1))
        p_text = self.W_p_text(F.normalize(text_emb + text_emb, dim=1))

        # Create user prompt from collaborative signals
        # Average multi-modal features
        item_feat_avg = (audio_emb + image_emb + text_emb) / 3.0

        # Aggregate via user-item matrix (Eq. 4)
        user_collab = torch.sparse.mm(user_item_mat, item_feat_avg)
        p_user = self.W_p_user(F.normalize(user_collab, dim=1))

        return p_audio, p_image, p_text, p_user, audio_emb, image_emb, text_emb


class TeacherModel(nn.Module):
    """GNN-based Teacher Model"""

    def __init__(self, n_users, n_items, embed_dim, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # ID embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Feature transformation with prompt guidance
        self.audio_transform = nn.Linear(embed_dim, embed_dim)
        self.image_transform = nn.Linear(embed_dim, embed_dim)
        self.text_transform = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_mat, audio_emb, image_emb, text_emb,
                p_audio, p_image, p_text, lambda_p=0.1):
        """
        Forward pass with GNN propagation
        """
        # Apply prompt guidance (Eq. 5)
        f_audio = self.audio_transform(audio_emb + lambda_p * p_audio)
        f_image = self.image_transform(image_emb + lambda_p * p_image)
        f_text = self.text_transform(text_emb + lambda_p * p_text)

        # Aggregate modality features
        item_feat = (f_audio + f_image + f_text) / 3.0

        # Initialize embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight + item_feat

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]

        # GNN propagation (Eq. 6)
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_mat, all_emb)
            embs.append(all_emb)

        # Aggregate all layers
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)

        user_final = final_emb[:self.n_users]
        item_final = final_emb[self.n_users:]

        return user_final, item_final, embs

    def get_modality_embs(self, audio_emb, image_emb, text_emb, adj_mat, user_item_mat):
        """Get modality-aware embeddings for MMC loss"""
        # Simple modality-specific embeddings
        user_emb = self.user_embedding.weight

        # User-modality embeddings through aggregation
        user_audio = torch.sparse.mm(user_item_mat, audio_emb)
        user_image = torch.sparse.mm(user_item_mat, image_emb)
        user_text = torch.sparse.mm(user_item_mat, text_emb)

        return {
            'audio': (user_audio, audio_emb),
            'image': (user_image, image_emb),
            'text': (user_text, text_emb)
        }


class StudentModel(nn.Module):
    """MLP-based Student Model"""

    def __init__(self, n_users, n_items, embed_dim, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # ID embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # MLP layers
        self.user_mlp = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
        self.item_mlp = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, audio_emb, image_emb, text_emb, teacher_embs=None):
        """
        Forward with residual connections to teacher (Eq. 7)
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight + (audio_emb + image_emb + text_emb) / 3.0

        user_embs = [user_emb]
        item_embs = [item_emb]

        for layer in range(self.n_layers):
            # MLP transformation
            user_h = torch.sigmoid(self.user_mlp[layer](user_embs[-1]))
            item_h = torch.sigmoid(self.item_mlp[layer](item_embs[-1]))

            # Add residual from teacher if available
            if teacher_embs is not None:
                teacher_layer = teacher_embs[:, layer, :]
                user_h = user_h + teacher_layer[:self.n_users]
                item_h = item_h + teacher_layer[self.n_users:]

            user_embs.append(user_h)
            item_embs.append(item_h)

        # Final embeddings
        user_final = user_embs[-1]
        item_final = item_embs[-1]

        return user_final, item_final, (user_embs, item_embs)


class PMMKD:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.device = config['device']

        # Initialize modules
        feat_dims = {
            'audio': dataset.audio_feat.shape[1],
            'image': dataset.image_feat.shape[1],
            'text': dataset.text_feat.shape[1]
        }

        self.prompt_module = PromptModule(
            dataset.n_users, dataset.n_items, feat_dims,
            config['embed_dim'], config['prompt_dim']
        ).to(self.device)

        self.teacher = TeacherModel(
            dataset.n_users, dataset.n_items,
            config['embed_dim'], config['n_layers']
        ).to(self.device)

        self.student = StudentModel(
            dataset.n_users, dataset.n_items,
            config['embed_dim'], config['n_layers']
        ).to(self.device)

        # Move data to device
        self.audio_feat = torch.FloatTensor(dataset.audio_feat).to(self.device)
        self.image_feat = torch.FloatTensor(dataset.image_feat).to(self.device)
        self.text_feat = torch.FloatTensor(dataset.text_feat).to(self.device)
        self.adj_mat = dataset.adj_mat.to(self.device)
        self.user_item_mat = dataset.user_item_mat.to(self.device)

        # Optimizers
        self.teacher_optimizer = Adam(
            list(self.prompt_module.parameters()) + list(self.teacher.parameters()),
            lr=config['lr']
        )

        self.student_optimizer = Adam(
            list(self.prompt_module.parameters()) + list(self.student.parameters()),
            lr=config['lr']
        )
        self.visualizer = ImprovedTSNEVisualizer(save_path='./visualizations/')

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        """BPR loss"""
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return loss

    def sample_batch(self, batch_size):
        """Sample training batch"""
        train_mat = self.dataset.train_mat
        users, pos_items, neg_items = [], [], []

        for _ in range(batch_size):
            user = np.random.randint(0, self.dataset.n_users)
            pos_items_user = train_mat[user].nonzero()[1]

            if len(pos_items_user) == 0:
                continue

            pos_item = np.random.choice(pos_items_user)

            neg_item = np.random.randint(0, self.dataset.n_items)
            while neg_item in pos_items_user:
                neg_item = np.random.randint(0, self.dataset.n_items)

            users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)

        return (torch.LongTensor(users).to(self.device),
                torch.LongTensor(pos_items).to(self.device),
                torch.LongTensor(neg_items).to(self.device))

    def train_teacher(self, epochs):
        """Stage 1: Train teacher model"""
        print("Stage 1: Training Teacher Model...")

        for epoch in range(epochs):
            self.teacher.train()
            self.prompt_module.train()

            total_loss = 0
            n_batch = self.dataset.train_mat.nnz // self.config['batch_size']

            for batch in tqdm(range(n_batch), desc=f"Teacher Epoch {epoch + 1}"):
                users, pos_items, neg_items = self.sample_batch(self.config['batch_size'])

                # Get prompts
                p_audio, p_image, p_text, p_user, audio_emb, image_emb, text_emb = \
                    self.prompt_module(self.audio_feat, self.image_feat, self.text_feat, self.adj_mat, self.user_item_mat)

                # Forward pass
                user_emb, item_emb, _ = self.teacher(
                    self.adj_mat, audio_emb, image_emb, text_emb,
                    p_audio, p_image, p_text, self.config['lambda_p']
                )

                # Get embeddings
                u_emb = user_emb[users]
                pos_emb = item_emb[pos_items]
                neg_emb = item_emb[neg_items]

                # BPR loss (Eq. 8)
                loss_rec = self.bpr_loss(u_emb, pos_emb, neg_emb)

                # Prompt loss (Eq. 10)
                u_prompt = p_user[users]
                pos_p = (p_audio[pos_items] + p_image[pos_items] + p_text[pos_items]) / 3.0
                neg_p = (p_audio[neg_items] + p_image[neg_items] + p_text[neg_items]) / 3.0
                loss_prompt = self.bpr_loss(u_prompt, pos_p, neg_p)

                # Total loss (Eq. 13)
                loss = loss_rec + self.config['lambda_p'] * loss_prompt

                self.teacher_optimizer.zero_grad()
                loss.backward()
                self.teacher_optimizer.step()

                total_loss += loss.item()

            print(f"Teacher Epoch {epoch + 1}, Loss: {total_loss / n_batch:.4f}")

    def train_student(self, epochs):
        """Stage 2: Train student with KD"""
        print("\nStage 2: Training Student Model with KD...")

        self.teacher.eval()

        for epoch in range(epochs):
            self.student.train()
            self.prompt_module.train()

            total_loss = 0
            n_batch = self.dataset.train_mat.nnz // self.config['batch_size']

            for batch in tqdm(range(n_batch), desc=f"Student Epoch {epoch + 1}"):
                users, pos_items, neg_items = self.sample_batch(self.config['batch_size'])

                # Get prompts
                p_audio, p_image, p_text, p_user, audio_emb, image_emb, text_emb = \
                    self.prompt_module(self.audio_feat, self.image_feat, self.text_feat, self.adj_mat, self.user_item_mat)

                # Teacher forward (no grad)
                with torch.no_grad():
                    user_emb_t, item_emb_t, teacher_embs = self.teacher(
                        self.adj_mat, audio_emb, image_emb, text_emb,
                        p_audio, p_image, p_text, self.config['lambda_p']
                    )

                # Student forward
                user_emb_s, item_emb_s, student_layer_embs = self.student(
                    audio_emb, image_emb, text_emb, teacher_embs
                )

                # Get embeddings
                u_emb_s = user_emb_s[users]
                pos_emb_s = item_emb_s[pos_items]
                neg_emb_s = item_emb_s[neg_items]

                u_emb_t = user_emb_t[users]
                pos_emb_t = item_emb_t[pos_items]
                neg_emb_t = item_emb_t[neg_items]

                # Student BPR loss (Eq. 18)
                loss_rec = self.bpr_loss(u_emb_s, pos_emb_s, neg_emb_s)

                # Inference-aligned KD (Eq. 14)
                pos_score_t = torch.sum(u_emb_t * pos_emb_t, dim=1)
                pos_score_s = torch.sum(u_emb_s * pos_emb_s, dim=1)
                neg_score_t = torch.sum(u_emb_t * neg_emb_t, dim=1)
                neg_score_s = torch.sum(u_emb_s * neg_emb_s, dim=1)

                loss_kd1 = F.kl_div(F.log_softmax(pos_score_s.unsqueeze(1), dim=1),
                                    F.softmax(pos_score_t.unsqueeze(1), dim=1), reduction='batchmean')
                loss_kd1 += F.kl_div(F.log_softmax(neg_score_s.unsqueeze(1), dim=1),
                                     F.softmax(neg_score_t.unsqueeze(1), dim=1), reduction='batchmean')

                # Embedding-aligned KD (Eq. 16) - simplified
                loss_kd3 = F.mse_loss(user_emb_s, user_emb_t) + F.mse_loss(item_emb_s, item_emb_t)

                # Total loss (Eq. 18)
                loss = (loss_rec +
                        self.config['lambda_1'] * loss_kd1 +
                        self.config['lambda_3'] * loss_kd3)

                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()

                total_loss += loss.item()

            print(f"Student Epoch {epoch + 1}, Loss: {total_loss / n_batch:.4f}")

            # Evaluate every few epochs
            if (epoch + 1) % 5 == 0:
                self.evaluate()

            if (epoch + 1) % 20 == 0:
                 model.visualize_user_item_joint(epoch=epoch+1, model_type='student')

            # if (epoch + 1) % 20 == 0:
            #     print(f"\nGenerating visualizations at epoch {epoch + 1}...")
            #     self.visualize_embeddings(
            #         save_path='./visualizations/student/',
            #         epoch=epoch + 1
            #     )

    def evaluate(self, K=20):
        """Evaluate student model"""
        self.student.eval()
        self.prompt_module.eval()

        with torch.no_grad():
            p_audio, p_image, p_text, p_user, audio_emb, image_emb, text_emb = \
                self.prompt_module(self.audio_feat, self.image_feat, self.text_feat, self.adj_mat, self.user_item_mat)

            user_emb, item_emb, _ = self.student(audio_emb, image_emb, text_emb)

            # Compute scores
            scores = torch.matmul(user_emb, item_emb.t())

            # Mask training items
            train_mat_tensor = torch.FloatTensor(self.dataset.train_mat.toarray()).to(self.device)
            scores = scores - train_mat_tensor * 1e10

            # Get top-K items
            _, topk_indices = torch.topk(scores, K, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            # Calculate metrics
            test_mat = self.dataset.test_mat.toarray()
            recalls, ndcgs = [], []

            for user in range(self.dataset.n_users):
                test_items = test_mat[user].nonzero()[0]
                if len(test_items) == 0:
                    continue

                pred_items = topk_indices[user]
                hits = np.intersect1d(test_items, pred_items)

                # Recall@K
                recall = len(hits) / min(len(test_items), K)
                recalls.append(recall)

                # NDCG@K
                dcg = sum([1.0 / np.log2(idx + 2) for idx, item in enumerate(pred_items) if item in test_items])
                idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(test_items), K))])
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)

            recall_avg = np.mean(recalls)
            ndcg_avg = np.mean(ndcgs)

            print(f"Recall@{K}: {recall_avg:.4f}, NDCG@{K}: {ndcg_avg:.4f}")

        return recall_avg, ndcg_avg

    def visualize_embeddings(self, save_path='./visualizations/', epoch=None)
        import os
        os.makedirs(save_path, exist_ok=True)

        self.student.eval()
        self.prompt_module.eval()

        with torch.no_grad():
            p_audio, p_image, p_text, p_user, audio_emb, image_emb, text_emb = \
                self.prompt_module(self.audio_feat, self.image_feat, self.text_feat,
                                   self.adj_mat, self.user_item_mat)

            user_emb, item_emb, _ = self.student(audio_emb, image_emb, text_emb)


            user_emb_np = user_emb.cpu().numpy()
            item_emb_np = item_emb.cpu().numpy()

        train_mat = self.dataset.train_mat
        user_activity = np.array(train_mat.sum(axis=1)).flatten()  # 每个用户的交互数
        item_popularity = np.array(train_mat.sum(axis=0)).flatten()  # 每个项目的交互数

        self._plot_tsne(
            user_emb_np,
            user_activity,
            title=f'User Embeddings t-SNE (Epoch {epoch})' if epoch else 'User Embeddings t-SNE',
            save_name=f'{save_path}/user_tsne_epoch_{epoch}.pdf' if epoch else f'{save_path}/user_tsne.pdf',
            colorbar_label='User Activity (# interactions)'
        )

        self._plot_tsne(
            item_emb_np,
            item_popularity,
            title=f'Item Embeddings t-SNE (Epoch {epoch})' if epoch else 'Item Embeddings t-SNE',
            save_name=f'{save_path}/item_tsne_epoch_{epoch}.pdf' if epoch else f'{save_path}/item_tsne.pdf',
            colorbar_label='Item Popularity (# interactions)'
        )

        self._plot_combined_tsne(
            user_emb_np,
            item_emb_np,
            title=f'User & Item Embeddings t-SNE (Epoch {epoch})' if epoch else 'User & Item Embeddings t-SNE',
            save_name=f'{save_path}/combined_tsne_epoch_{epoch}.pdf' if epoch else f'{save_path}/combined_tsne.pdf'
        )

        print(f"Visualizations saved to {save_path}")

    def _plot_tsne(self, embeddings, colors, title, save_name, colorbar_label, n_samples=2000, perplexity=30):
        if embeddings.shape[0] > n_samples:
            indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
            embeddings_sample = embeddings[indices]
            colors_sample = colors[indices]
        else:
            embeddings_sample = embeddings
            colors_sample = colors

        print(f"Running t-SNE on {embeddings_sample.shape[0]} samples...")

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, embeddings_sample.shape[0] - 1))
        embeddings_2d = tsne.fit_transform(embeddings_sample)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors_sample,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, label=colorbar_label)
        plt.title(title, fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_name, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")

    def _plot_combined_tsne(self, user_emb, item_emb, title, save_name, n_samples=1000):
        if user_emb.shape[0] > n_samples:
            user_indices = np.random.choice(user_emb.shape[0], n_samples, replace=False)
            user_sample = user_emb[user_indices]
        else:
            user_sample = user_emb

        if item_emb.shape[0] > n_samples:
            item_indices = np.random.choice(item_emb.shape[0], n_samples, replace=False)
            item_sample = item_emb[item_indices]
        else:
            item_sample = item_emb

        combined_emb = np.vstack([user_sample, item_sample])
        labels = np.array(['User'] * user_sample.shape[0] + ['Item'] * item_sample.shape[0])

        print(f"Running t-SNE on {combined_emb.shape[0]} samples (users + items)...")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(combined_emb)

        plt.figure(figsize=(12, 10))

        user_mask = labels == 'User'
        item_mask = labels == 'Item'

        plt.scatter(
            embeddings_2d[user_mask, 0],
            embeddings_2d[user_mask, 1],
            c='blue',
            label='Users',
            alpha=0.6,
            s=20
        )
        plt.scatter(
            embeddings_2d[item_mask, 0],
            embeddings_2d[item_mask, 1],
            c='red',
            label='Items',
            alpha=0.6,
            s=20
        )

        plt.title(title, fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_name, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")

    def visualize_modality_embeddings(self, save_path='./visualizations/'):
        import os
        os.makedirs(save_path, exist_ok=True)

        self.prompt_module.eval()

        with torch.no_grad():
            p_audio, p_image, p_text, p_user, audio_emb, image_emb, text_emb = \
                self.prompt_module(self.audio_feat, self.image_feat, self.text_feat,
                                   self.adj_mat, self.user_item_mat)

            audio_emb_np = audio_emb.cpu().numpy()
            image_emb_np = image_emb.cpu().numpy()
            text_emb_np = text_emb.cpu().numpy()

        n_samples = min(1000, audio_emb_np.shape[0])
        indices = np.random.choice(audio_emb_np.shape[0], n_samples, replace=False)

        combined_emb = np.vstack([
            audio_emb_np[indices],
            image_emb_np[indices],
            text_emb_np[indices]
        ])

        modality_labels = np.array(
            ['Audio'] * n_samples +
            ['Image'] * n_samples +
            ['Text'] * n_samples
        )

        print(f"Running t-SNE on {combined_emb.shape[0]} modality embeddings...")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(combined_emb)

        plt.figure(figsize=(12, 10))

        colors = {'Audio': 'red', 'Image': 'green', 'Text': 'blue'}
        for modality in ['Audio', 'Image', 'Text']:
            mask = modality_labels == modality
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[modality],
                label=modality,
                alpha=0.6,
                s=20
            )

        plt.title('Multi-Modal Item Embeddings t-SNE', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_path}/modality_tsne.pdf', dpi=600, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}/modality_tsne.pdf")

    def visualize_user_item_joint(self, epoch=None, model_type='student'):
        print(f"\n{'=' * 70}")
        print(f"Visualizing User-Item Joint Embeddings ({model_type.upper()} Model)")
        print(f"{'=' * 70}\n")

        save_path = f'./visualizations/epoch_{epoch}/' if epoch else './visualizations/final/'
        visualizer = ImprovedTSNEVisualizer(save_path=save_path)

        if model_type == 'student':
            self.student.eval()
        else:
            self.teacher.eval()
        self.prompt_module.eval()

        with torch.no_grad():
            p_audio, p_image, p_text, p_user, audio_emb, image_emb, text_emb = \
                self.prompt_module(self.audio_feat, self.image_feat, self.text_feat,
                                   self.adj_mat, self.user_item_mat)

            if model_type == 'student':
                user_emb, item_emb, _ = self.student(audio_emb, image_emb, text_emb)
            else:
                user_emb, item_emb, _ = self.teacher(
                    self.adj_mat, audio_emb, image_emb, text_emb,
                    p_audio, p_image, p_text, self.config['lambda_p']
                )

            user_emb_np = user_emb.cpu().numpy()
            item_emb_np = item_emb.cpu().numpy()

        train_mat = self.dataset.train_mat.toarray() if hasattr(self.dataset.train_mat,
                                                                'toarray') else self.dataset.train_mat
        user_activity = np.array(train_mat.sum(axis=1)).flatten()
        item_popularity = np.array(train_mat.sum(axis=0)).flatten()

        metrics = visualizer.plot_user_item_joint(
            user_emb=user_emb_np,
            item_emb=item_emb_np,
            interaction_mat=train_mat,
            user_activity=user_activity,
            item_popularity=item_popularity,
            n_samples_user=1000,
            n_samples_item=1000,
            show_interactions=True,
            n_interactions=50,
            name=f'{model_type}_model'
        )

        return metrics
