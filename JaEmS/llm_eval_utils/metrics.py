import torch


def baseline_consistency(s, baseline, similarity):
    return torch.max(similarity(s, baseline), 1)[0]

    
def self_consistency(s, S, similarity):
    return baseline_consistency(s, torch.mean(S, 0), similarity)


def human_consistency(s, human_baseline, similarity):
    return baseline_consistency(s, human_baseline, similarity)


def MAHC(s, human_baseline, similarity):
    return torch.mode(torch.max(similarity(s, baseline))[1])
