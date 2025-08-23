import torch
import torch.nn as nn
from addict import Dict
from torch_geometric.nn import GATv2Conv

from .transformers import TransformerMLP


def batch_graphs(x_list, edge_index_list):
    """
    x_list: list of (M+N, D) node feature tensors (length B)
    edge_index_list: list of (2, E_i) edge indices for each graph
    Returns:
        x_all:       (total_nodes, D)
        edge_index:  (2, total_edges)
        batch:       (total_nodes,) → maps node idx to graph idx
        graph_ptrs:  list of start indices for each graph in x_all (for slicing if needed)
    """
    x_all = []
    edge_all = []
    batch = []
    graph_ptrs = []

    node_offset = 0
    for i, (x, edge_index) in enumerate(zip(x_list, edge_index_list)):
        num_nodes = x.size(0)

        edge_all.append(edge_index + node_offset)

        x_all.append(x)

        batch.append(torch.full((num_nodes,), i, dtype=torch.long, device=x.device))

        graph_ptrs.append(node_offset)
        node_offset += num_nodes

    x_all = torch.cat(x_all, dim=0)
    edge_index = torch.cat(edge_all, dim=1)
    batch = torch.cat(batch, dim=0)

    return x_all, edge_index, batch, graph_ptrs


def unbatch_tensor(x_all, graph_ptrs, sizes):
    """
    Split a batched tensor using the graph_ptrs and node counts.

    x_all:     (total_nodes, D)
    graph_ptrs: list of start indices (from batch_graphs)
    sizes:     list of node counts per graph
    Returns:
        List of tensors per graph
    """
    return [x_all[start : start + size] for start, size in zip(graph_ptrs, sizes)]


class GraphormerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        heads=4,
        dropout=0.1,
        is_final_block=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        dropout if self.training else 0
        out_channels = n_embd if is_final_block else n_embd // heads
        concat = False if is_final_block else True
        self.norm1 = nn.LayerNorm(n_embd)
        self.gat = GATv2Conv(
            in_channels=n_embd,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=False,
        )

        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = TransformerMLP(
            Dict(
                n_embd=n_embd,
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index):
        # x: (N, n_embd), edge_index: (2, E)
        x_res = x
        x = self.norm1(x)
        x = self.gat(x, edge_index)
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res

        return x
