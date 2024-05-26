import torch

from graphormer.modules.encoding import EdgeEncoding
from graphormer.modules.model_data import ModelData


class TestEdgeEncodingGroup:
    def test_edge_encoding(self):
        torch.set_grad_enabled(False)
        batch_size = 2
        edge_embedding_dim = 3
        max_path_distance = 4
        num_nodes = 5
        edge_encoding = EdgeEncoding(edge_embedding_dim, max_path_distance)

        edge_embedding = torch.ones(batch_size, num_nodes, edge_embedding_dim) * -1
        edge_embedding[0] = torch.Tensor(
            [
                [0.0, 1.0, 2.0],
                [6.0, 5.0, 4.0],
                [0.0, 1.0, 2.0],
                [0.0, -1.0, -2.0],
                [-6.0, -5.0, -4.0],
            ]
        )
        edge_embedding[1] = torch.Tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
        )

        edge_paths = torch.ones(batch_size, num_nodes**2, max_path_distance) * -1
        edge_paths[0, :5] = torch.Tensor(
            [
                [0, 1, 2, 3],
                [3, 2, 1, -1],
                [1, 2, -1, -1],
                [2, -1, -1, -1],
                [-1, -1, -1, -1],
            ]
        )
        edge_paths[1, 5:10] = torch.Tensor(
            [
                [
                    [0, -1, -1, -1],
                    [0, 1, -1, -1],
                    [1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ]
            ]
        )
        edge_paths = edge_paths.long()

        device = torch.device("cpu")
        data = ModelData(torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), edge_paths, device)
        data.edge_embedding = edge_embedding
        data: ModelData = edge_encoding(data)
        assert data.edge_encoding is not None

        assert data.edge_encoding.shape == (batch_size, num_nodes**2)
        # encoding in batch 0 of node 0 -> 0 by edges
        # x_e_n - the feature of the nth edge in the path
        # (edge_embedding_dim)
        x_e_0 = edge_embedding[0, edge_paths[0, 0, 0]]
        x_e_1 = edge_embedding[0, edge_paths[0, 0, 1]]
        x_e_2 = edge_embedding[0, edge_paths[0, 0, 2]]
        x_e_3 = edge_embedding[0, edge_paths[0, 0, 3]]

        # w_n - the nth weight embedding
        # (edge_embedding_dim)
        w_0 = edge_encoding.edge_vector[0]
        w_1 = edge_encoding.edge_vector[1]
        w_2 = edge_encoding.edge_vector[2]
        w_3 = edge_encoding.edge_vector[3]

        # sum x_e_n dot w_n
        sum_n = torch.dot(x_e_0, w_0) + torch.dot(x_e_1, w_1) + torch.dot(x_e_2, w_2) + torch.dot(x_e_3, w_3)
        expected_encoding = sum_n / 4

        assert data.edge_encoding[0, 0] == expected_encoding

        # Same as above, but our path length is only 3
        x_e_0 = edge_embedding[0, edge_paths[0, 1, 0]]
        x_e_1 = edge_embedding[0, edge_paths[0, 1, 1]]
        x_e_2 = edge_embedding[0, edge_paths[0, 1, 2]]

        sum_n = torch.dot(x_e_0, w_0) + torch.dot(x_e_1, w_1) + torch.dot(x_e_2, w_2)
        expected_encoding = sum_n / 3

        assert data.edge_encoding[0, 1] == expected_encoding

        # Same as above, but our path length is only 2
        x_e_0 = edge_embedding[0, edge_paths[0, 2, 0]]
        x_e_1 = edge_embedding[0, edge_paths[0, 2, 1]]

        sum_n = torch.dot(x_e_0, w_0) + torch.dot(x_e_1, w_1)
        expected_encoding = sum_n / 2

        assert data.edge_encoding[0, 2] == expected_encoding

        # Same as above, but our path length is only 1
        x_e_0 = edge_embedding[0, edge_paths[0, 3, 0]]

        sum_n = torch.dot(x_e_0, w_0)
        expected_encoding = sum_n

        assert data.edge_encoding[0, 3] == expected_encoding
        assert data.edge_encoding[0, 4] == torch.Tensor([0])

        # Now check for the second batch
        x_e_0 = edge_embedding[1, edge_paths[1, 5, 0]]

        sum_n = torch.dot(x_e_0, w_0)
        expected_encoding = sum_n

        assert data.edge_encoding[1, 5] == expected_encoding

        x_e_0 = edge_embedding[1, edge_paths[1, 6, 0]]
        x_e_1 = edge_embedding[1, edge_paths[1, 6, 1]]

        sum_n = torch.dot(x_e_0, w_0) + torch.dot(x_e_1, w_1)
        expected_encoding = sum_n / 2
        assert data.edge_encoding[1, 6] == expected_encoding

        x_e_0 = edge_embedding[1, edge_paths[1, 7, 0]]

        sum_n = torch.dot(x_e_0, w_0)
        expected_encoding = sum_n

        assert data.edge_encoding[1, 7] == expected_encoding

        assert (data.edge_encoding[0, 5:] == 0).all(), data.edge_encoding
        assert (data.edge_encoding[1, :5] == 0).all(), data.edge_encoding
        assert (data.edge_encoding[1, 8:] == 0).all(), data.edge_encoding
