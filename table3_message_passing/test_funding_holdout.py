import numpy as np

from dataset import positions
from funding_holdout import holdout_graphs
from test_smoke import graphs


def test_held_out_funding_removed_across_roles(graphs):
    parts = tuple(np.array(rows) for rows in ([1, 2], [3], [0, 4, 5]))
    held, audit = holdout_graphs(graphs, parts)
    assert audit["test_funding_clusters"] == 1
    assert audit["test_funding_clusters_in_fit_or_validation"] == 0
    assert audit["fit"]["removed_node_instances"] == 1
    assert audit["fit"]["removed_edges"] == 1
    assert audit["validation"]["removed_node_instances"] == 0
    assert audit["test"]["removed_node_instances"] == 0
    node_rows, _, _ = positions(held.node_ptr, np.array([1]))
    assert held.entity_ids[held.entity_rows[node_rows]].tolist() == [21]
    assert held.edge_ptr[2] == held.edge_ptr[1]
