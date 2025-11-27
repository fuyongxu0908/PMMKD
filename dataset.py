import pickle
import numpy as np
import torch


class TikTokDataset:
    def __init__(self, data_path='./data/tiktok/'):
        self.data_path = data_path

        # Load interaction matrices
        with open(os.path.join(data_path, 'trnMat.pkl'), 'rb') as f:
            train_mat_loaded = pickle.load(f)
            # Convert to CSR for efficient row access
            self.train_mat = train_mat_loaded.tocsr() if hasattr(train_mat_loaded, 'tocsr') else train_mat_loaded

        with open(os.path.join(data_path, 'valMat.pkl'), 'rb') as f:
            val_mat_loaded = pickle.load(f)
            self.val_mat = val_mat_loaded.tocsr() if hasattr(val_mat_loaded, 'tocsr') else val_mat_loaded

        with open(os.path.join(data_path, 'tstMat.pkl'), 'rb') as f:
            test_mat_loaded = pickle.load(f)
            self.test_mat = test_mat_loaded.tocsr() if hasattr(test_mat_loaded, 'tocsr') else test_mat_loaded

        # Load features
        self.audio_feat = np.load(os.path.join(data_path, 'audio_feat.npy'))
        self.image_feat = np.load(os.path.join(data_path, 'image_feat.npy'))
        self.text_feat = np.load(os.path.join(data_path, 'text_feat.npy'))

        # Get dimensions
        self.n_users = self.train_mat.shape[0]
        self.n_items = self.train_mat.shape[1]

        # Create adjacency matrix
        self.adj_mat, self.user_item_mat = self._create_adj_mat()

        print(f"Users: {self.n_users}, Items: {self.n_items}")
        print(f"Audio feat dim: {self.audio_feat.shape[1]}")
        print(f"Image feat dim: {self.image_feat.shape[1]}")
        print(f"Text feat dim: {self.text_feat.shape[1]}")

    def _create_adj_mat(self):
        """Create normalized adjacency matrix"""
        import scipy.sparse as sp

        # Convert to COO format
        train_coo = sp.coo_matrix(self.train_mat)

        # Build adjacency matrix
        adj_mat = sp.dok_matrix((self.n_users + self.n_items,
                                 self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # Add edges
        R = train_coo.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        # Normalize
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()

        # Convert to torch sparse tensor
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        adj_sparse = torch.sparse.FloatTensor(indices, values, shape)

        # Create normalized user-item submatrix for prompt module
        # Extract diagonal values for users and items
        d_inv_sqrt_array = d_inv_sqrt  # This is already a numpy array
        d_user = d_inv_sqrt_array[:self.n_users]
        d_item = d_inv_sqrt_array[self.n_users:]

        # Create separate diagonal matrices
        d_mat_user = sp.diags(d_user)
        d_mat_item = sp.diags(d_item)

        # Normalize user-item matrix: D_u^(-1/2) * R * D_i^(-1/2)
        user_item_norm = d_mat_user.dot(R).dot(d_mat_item).tocoo()

        ui_indices = torch.LongTensor(np.vstack([user_item_norm.row, user_item_norm.col]))
        ui_values = torch.FloatTensor(user_item_norm.data)
        ui_shape = torch.Size([self.n_users, self.n_items])
        user_item_sparse = torch.sparse.FloatTensor(ui_indices, ui_values, ui_shape)

        return adj_sparse, user_item_sparse

