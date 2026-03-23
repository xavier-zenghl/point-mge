import torch
from tools.fewshot import run_fewshot_episode

def test_fewshot_episode():
    support_features = torch.randn(50, 128)
    support_labels = torch.arange(5).repeat_interleave(10)
    query_features = torch.randn(50, 128)
    query_labels = torch.arange(5).repeat_interleave(10)
    acc = run_fewshot_episode(support_features, support_labels, query_features, query_labels)
    assert 0 <= acc <= 100

def test_fewshot_perfect_classification():
    support_features = torch.zeros(10, 2)
    support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    support_features[:5, 0] = 10.0
    support_features[5:, 1] = 10.0
    query_features = torch.zeros(4, 2)
    query_labels = torch.tensor([0, 0, 1, 1])
    query_features[:2, 0] = 10.0
    query_features[2:, 1] = 10.0
    acc = run_fewshot_episode(support_features, support_labels, query_features, query_labels)
    assert acc == 100.0
