import torch
import numpy

# # Hand crafted "semantic" strategy for Traffic Sign dataset
# strat1 = {
#     0: 8,
#     1: 8,
#     2: 8,
#     3: 8,
#     4: 8,
#     5: 8,
#     6: 8,
#     7: 8,
#     8: 0,
#     9: 10,
#     10: 9,
#     11: 11,
#     12: 13,
#     13: 12,
#     14: 7,
#     15: 16,
#     16: 15,
#     17: 13,
#     18: 18,
#     19: 20,
#     20: 19,
#     21: 21,
#     22: 22,
#     23: 23,
#     24: 24,
#     25: 25,
#     26: 26,
#     27: 27,
#     28: 8,
#     29: 29,
#     30: 30,
#     31: 31,
#     32: 32,
#     33: 34,
#     34: 33,
#     35: 35,
#     36: 37,
#     37: 36,
#     38: 39,
#     39: 38,
#     40: 40,
#     41: 41,
#     42: 42
# }

# int to strategy name dict
training_strategy_dict = {
    0: 'vanilla',
    1: 'worst_case',
    2: 'oracle',
    3: 'noisy_oracle',
    4: 'multi_targets',
    5: 'noisy_subset_oracle'
}

# # dictionary for predefined strategies
# strat_dict = {
#     'strat1': strat1
# }


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
