
"""Directed message passing with one logit per observation graph."""
import torch
from torch import nn
from torch.nn import functional as F

INPUT_WIDTH = 46
WIDTH = 32
HEADS = 4
HEAD_WIDTH = WIDTH // HEADS
GIN_LAYERS = 4
MODELS = ("gcn", "graphsage", "gat", "gin")


def scatter_sum(values, index, size):
    result = values.new_zeros((size, *values.shape[1:]))
    return result.index_add_(0, index, values)


def readout(x, batch, mean=False):
    pooled = scatter_sum(x, batch.graph, len(batch.sizes))
    return pooled / batch.sizes[:, None] if mean else pooled


class GCNLayer(nn.Module):
    def __init__(self, in_width):
        super().__init__()
        self.linear = nn.Linear(in_width, WIDTH, bias=False)
        self.bias = nn.Parameter(torch.zeros(WIDTH))

    def forward(self, x, src, dst, weights):
        projected = self.linear(x)
        messages = projected[src] * weights[:, None]
        return scatter_sum(messages, dst, len(x)) + self.bias


class SAGELayer(nn.Module):
    def __init__(self, in_width):
        super().__init__()
        self.linear = nn.Linear(2 * in_width, WIDTH)

    def forward(self, x, src, dst, degree):
        neighbors = scatter_sum(x[src], dst, len(x)) / degree[:, None]
        return F.normalize(self.linear(torch.cat((x, neighbors), dim=1)).relu(), dim=1)


class GATLayer(nn.Module):
    def __init__(self, in_width):
        super().__init__()
        self.linear = nn.Linear(in_width, WIDTH, bias=False)
        self.source_attention = nn.Parameter(torch.empty(HEADS, HEAD_WIDTH))
        self.target_attention = nn.Parameter(torch.empty(HEADS, HEAD_WIDTH))
        self.bias = nn.Parameter(torch.zeros(WIDTH))
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, x, src, dst):
        projected = self.linear(x).reshape(-1, HEADS, HEAD_WIDTH)
        sender = (projected * self.source_attention).sum(dim=-1)
        receiver = (projected * self.target_attention).sum(dim=-1)
        scores = F.leaky_relu(sender[src] + receiver[dst], negative_slope=0.2)
        dst_heads = dst[:, None].expand(-1, HEADS)
        maxima = scores.new_full((len(x), HEADS), -torch.inf)
        maxima.scatter_reduce_(0, dst_heads, scores.detach(), reduce="amax")
        exp = (scores - maxima[dst]).exp()
        attention = exp / scatter_sum(exp, dst, len(x))[dst]
        result = scatter_sum(projected[src] * attention[:, :, None], dst, len(x))
        return result.reshape(-1, WIDTH) + self.bias


class GINLayer(nn.Module):
    def __init__(self, in_width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_width, WIDTH), nn.BatchNorm1d(WIDTH),
                                 nn.ReLU(), nn.Linear(WIDTH, WIDTH))
        self.norm = nn.BatchNorm1d(WIDTH)

    def forward(self, x, src, dst):
        return self.norm(self.mlp(x + scatter_sum(x[src], dst, len(x)))).relu()


class GraphClassifier(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")
        self.name = name
        layer = {"gcn": GCNLayer, "graphsage": SAGELayer, "gat": GATLayer, "gin": GINLayer}[name]
        depth = GIN_LAYERS if name == "gin" else 2
        self.layers = nn.ModuleList(layer(INPUT_WIDTH if i == 0 else WIDTH) for i in range(depth))
        if name == "gin":
            self.heads = nn.ModuleList(nn.Linear(width, 1) for width in [INPUT_WIDTH] + [WIDTH] * depth)
            self.dropout = nn.Dropout(0.5)
        elif name == "gcn":
            self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 1))
        else:
            self.head = nn.Linear(WIDTH, 1)

    def forward(self, batch):
        x = batch.x
        src, dst = batch.edges
        if self.name == "gin":
            logits = self.dropout(self.heads[0](readout(x, batch)))
            for layer, head in zip(self.layers, self.heads[1:]):
                x = layer(x, src, dst)
                logits = logits + self.dropout(head(readout(x, batch)))
            return logits.squeeze(-1)

        if self.name in ("gcn", "gat"):
            self_nodes = torch.arange(len(x), device=x.device)
            src, dst = torch.cat((src, self_nodes)), torch.cat((dst, self_nodes))
        if self.name == "gcn":
            degree = scatter_sum(x.new_ones(len(dst)), dst, len(x)).rsqrt()
            weights = degree[src] * degree[dst]
            for layer in self.layers:
                x = layer(x, src, dst, weights).relu()
        elif self.name == "graphsage":
            degree = scatter_sum(x.new_ones(len(dst)), dst, len(x)).clamp_min_(1)
            for layer in self.layers:
                x = layer(x, src, dst, degree)
        else:
            for layer in self.layers:
                x = F.elu(layer(x, src, dst))
        return self.head(readout(x, batch, mean=True)).squeeze(-1)
