import torch
import numpy


# int to strategy name dict
training_strategy_dict = {
    0: 'vanilla',
    1: 'worst_case',
    2: 'oracle',
    3: 'noisy_oracle',
    4: 'multi_targets',
    5: 'noisy_subset_oracle'
}


# Generate column strategy dict
def get_col_strategy(n_classes, target):
    col_strat = {}
    for source in range(n_classes):
        col_strat[source] = (target, 1)
    return col_strat


# Generate random strategy dict
def get_random_strategy(n_classes=43):
    idxs = torch.randint(low=0, high=n_classes, size=(n_classes,))
    budget = torch.ones_like(idxs) # TODO: remove if remains unsused
    strat = {}
    for source, target in enumerate(idxs):
        strat[source] = (target.item(), budget[source])
    return strat


# Generate random strategy dict
def get_random_subset_strategy(n_classes, targets_mat):
    strat = {}
    for source in range(n_classes):
        targets = targets_mat[source].nonzero().squeeze(1)
        num_targets = len(targets)
        rand_target_idx = torch.randint(low=0, high=num_targets, size=(1,)).item()
        rand_target = targets[rand_target_idx]
        strat[source] = (rand_target.item(), 1) # target, budget
    return strat
    
    
# Create torch target class tensor given gt labels and strategy dictionary
def translate_label_by_strategy(labels, strat):
    labels_arr = []
    budget_arr = []
    for label in labels:
        labels_arr.append(strat[label.item()][0])
        budget_arr.append(strat[label.item()][1])
    return torch.tensor(labels_arr, device=labels.device), torch.tensor(budget_arr, device=labels.device)


def translate_multi_labels_by_strategy(labels, strat):
    labels_mat = torch.tensor([]).bool()
    for label in labels:
        labels_mat = torch.concat((labels_mat, strat[label.item()].unsqueeze(0)))
    return labels_mat.to(labels.device)
