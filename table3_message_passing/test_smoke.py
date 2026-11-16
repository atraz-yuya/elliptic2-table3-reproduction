
import numpy as np
import pandas as pd
import pytest
import torch

from dataset import FEATURES, pack
from models import GATLayer, GCNLayer, GraphClassifier, MODELS, SAGELayer, readout
from run import evaluate, fit_model, threshold_for


@pytest.fixture
def graphs():
    labels = pd.DataFrame({"ccId": [10, 20, 30, 40, 50, 60], "y": [0, 1, 0, 1, 0, 1]})
    members = pd.DataFrame({"ccId": [10, 10, 20, 30, 40, 50, 60],
                            "clId": [11, 12, 21, 31, 41, 51, 61]})
    internal = pd.DataFrame({"src": [11, 11, 12], "dst": [12, 12, 11]})
    boundary = pd.DataFrame({"node_id": [11, 11, 21, 41], "direction": [0, 1, 1, 0],
                             "neighbor_id": [90, 90, 90, 91]})
    ids = [11, 12, 21, 31, 41, 51, 61, 90, 91]
    attrs = pd.DataFrame(np.tile(ids, (43, 1)).T, index=ids, columns=FEATURES)
    attrs["feat#2"] = 4
    return pack(labels, members, internal, boundary, attrs)


def test_observation_graph_and_fold_scaling(graphs):
    inventory = graphs.inventory()
    assert inventory.edges.tolist() == [4, 1, 0, 1, 0, 0]
    assert inventory.dual_role.tolist() == [1, 0, 0, 0, 0, 0]
    mean, scale = graphs.scaler([0, 1])
    assert mean[0] == np.mean([11, 12, 21, 90])
    assert (mean[1], scale[1]) == (4, 1)
    assert graphs.overlap([0], [1, 2])["shared_graphs"] == 1
    batch = graphs.batch(np.array([0, 1, 2]), graphs.features, "cpu")
    assert batch.sizes.tolist() == [3, 2, 1]
    assert set(map(tuple, batch.edges.T.tolist())) == {(0, 1), (1, 0), (2, 0), (0, 2), (3, 4)}
    assert batch.x[:, 43:].tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 1],
                                       [1, 0, 0], [0, 0, 1], [1, 0, 0]]
    assert readout(batch.x[:, :1], batch).flatten().tolist() == [113, 111, 31]
    assert torch.allclose(readout(batch.x[:, :1], batch, mean=True).flatten(),
                          torch.tensor([113 / 3, 111 / 2, 31]))


def test_directed_aggregation():
    x = torch.tensor([[2.0], [6.0], [10.0]])
    src, dst = torch.tensor([0]), torch.tensor([1])
    sage = SAGELayer(1)
    with torch.no_grad():
        sage.linear.bias.zero_()
        sage.linear.weight.zero_()
        sage.linear.weight[0, 0] = 1
        sage.linear.weight[1, 1] = 1
    result = sage(x, src, dst, torch.ones(3))[:, :2]
    expected = torch.nn.functional.normalize(torch.tensor([[2., 0.], [6., 2.], [10., 0.]]), dim=1)
    torch.testing.assert_close(result, expected)

    gcn = GCNLayer(1)
    gat = GATLayer(1)
    with torch.no_grad():
        gcn.linear.weight.fill_(1)
        gat.linear.weight.fill_(1)
        gat.source_attention.zero_()
        gat.target_attention.zero_()
    src, dst = torch.tensor([0, 0, 1, 2]), torch.tensor([1, 0, 1, 2])
    degree = torch.tensor([1., 2., 1.]).rsqrt()
    torch.testing.assert_close(gcn(x, src, dst, degree[src] * degree[dst])[:, 0],
                               torch.tensor([2., 3. + 2 ** 0.5, 10.]))
    torch.testing.assert_close(gat(x, src, dst)[:, 0], torch.tensor([2., 4., 10.]))


@pytest.mark.parametrize("name", MODELS)
def test_full_cpu_path(name, graphs, tmp_path):
    device = torch.device("cpu")
    torch.set_num_threads(1)
    batch = graphs.batch(np.array([0, 1, 2]), graphs.features, device)
    model = GraphClassifier(name)
    model.eval()
    before = model(batch).detach()
    assert before.shape == (3,)
    for row in range(3):
        separate = graphs.batch(np.array([row]), graphs.features, device)
        torch.testing.assert_close(model(separate).detach(), before[row:row + 1])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = [p.detach().clone() for p in model.parameters()]
    torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch.y).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), params))
    fit_model(name, 1, graphs, tuple(np.array(r) for r in ([0, 1], [2, 3], [4, 5])),
              device, tmp_path, steps=2, interval=1, batch_size=2)
    predictions = pd.read_csv(tmp_path / "outer_predictions.csv")
    assert predictions.ccId.tolist() == [50, 60]
    assert predictions.score.between(0, 1).all()
    assert (tmp_path / "metrics.json").is_file()


def test_threshold_tie_and_inclusion():
    y = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    threshold, f1 = threshold_for(y, scores)
    assert threshold == 0.9
    metrics, decisions = evaluate(y, scores, threshold)
    assert metrics["f1"] == pytest.approx(f1)
    assert decisions.tolist() == [True, False, False, False]
