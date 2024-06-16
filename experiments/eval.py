import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from datetime import datetime
import os
import sys
import pickle
# add import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_utils import get_dataset
from utils import translate_multi_labels_by_strategy
from attacks.util_fns import *
from attacks.attacks import PGDAttack, LinfPGDAttack, L2PGDAttack
from models.ViT import ViT

from models.model_zoo import PreActResNet
from models.vgg import vgg11_bn

# Globals
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    #parser.add_argument('--output_dir', type=str, default='output_dir', help='path for saving checkpoints')
    # Attack HPs
    parser.add_argument('--ord', type=int, default=0, help="0 for Linf, 1 for L2")
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--nb_iter', type=int, default=20)
    parser.add_argument('--eps_iter_factor', type=float, default=1.0)
    # Testing strategy
    #parser.add_argument('--testing_strategy', type=str, default='random')
    parser.add_argument("--use_aug", type=int, default=1, help="Use data augmentations")
    ###parser.add_argument('--test_loss', type=int, default=0, help="0: CrossEntropy 1: logit gt minus target 2: max minus target")
    # Data
    parser.add_argument('--dataset', type=str, default='bazyl/GTSRB')
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--class_list', nargs='*', type=int, default=None, help="pass partial class list")
    # model
    parser.add_argument('--arch', type=str, default='vgg11_bn')
    parser.add_argument('--model_path', type=str, default=None)
    # Whether the attacker stops when fooled and not attacking when not fooled
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='.')
    # tests
    parser.add_argument('--test_loss_list', nargs='*', type=int, default=[], help="list of targeted test losses")
    parser.add_argument('--mt_loss_list', nargs='*', type=int, default=[], help="list of multi-target test losses")
    parser.add_argument('--targets_mat_path', type=str, default=None, help="path to a pickled binary mat describing each source's targets subset. only applicable to multi targets test")
    parser.add_argument('--omit_clean', action='store_true', help="omit clean evaluation")
    parser.add_argument('--omit_adv', action='store_true', help="omit adv evaluation")
    parser.add_argument('--log_preds', action='store_true', help="log predictions in csv format")

    args, _ = parser.parse_known_args()

    # additional arguments    
    args.n_classes = len(args.class_list) if args.class_list is not None else 43 if args.dataset in ['bazyl/GTSRB', 'GTSRB'] else 10
    args.targets_mat = None
    if args.targets_mat_path is not None:
        args.targets_mat = pickle.load(open(args.targets_mat_path, 'rb'))
    args.gpu_ids = None if device == 'cpu' else [i for i in range(args.num_gpus)]
    print(args.gpu_ids)
    return args


def load_model(arch_name, n_classes, dataset, model_path):
    model = None
    if arch_name == 'resnet':
        model = PreActResNet(n_classes, dataset=dataset).to(device)
    elif arch_name == 'vgg11_bn': # vgg11_bn
        model = vgg11_bn(n_classes, dataset=dataset).to(device)
    elif arch_name == 'ViT':
        model = ViT(n_classes, dataset=dataset).to(device)
    else:
        raise Exception(f"Illegal arch: {arch_name}")
    try:
        m, u = model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        assert len(m) == 0 and len(u) == 0
    except RuntimeError:
        print("Encountered RuntimeError while loading state_dict. Trying to remove module from all keys.")
        orig_state_dict = torch.load(model_path, map_location='cuda:0')
        state_dict = {key.replace("module.", ""): value for key, value in orig_state_dict.items()}
        m, u = model.load_state_dict(state_dict)
        assert len(m) == 0 and len(u) == 0
    return model


def get_data_loader(dataset_name, set_type, batch_size, image_size, class_list = None):
    train_set = None
    test_set = None
    print("loading", dataset_name, "dataset")
    if 'GTSRB' in dataset_name:
        train_set, test_set = get_dataset(data='bazyl/GTSRB', image_size=image_size, class_list=class_list)
    else: # cifar10
        train_set, test_set = get_dataset(data=dataset_name, image_size=image_size, class_list=class_list)
    requested_set = train_set if set_type == "train" else test_set
    labels = requested_set.tensors[1]
    class_counts = torch.bincount(labels, weights=None)
    data_loader = torch.utils.data.DataLoader(requested_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return data_loader, class_counts



def create_attck(model, is_targeted, loss_num, params):
    attack_fac = LinfPGDAttack if params["ord"] == np.inf else L2PGDAttack
    return attack_fac(predict=model, eps=params["eps"], nb_iter=params["nb_iter"],
                      eps_iter=params["eps_iter"],
                      targeted=is_targeted,
                      loss_fn=loss_num)


def evaluate_adv(model, attack, test_loader, path=None, log_preds=False):
    print("evaluating adversarial non targeted attack")
    model.eval()
    count_correct = 0
    n_images = 0
    preds_df = pd.DataFrame()
    for batch_idx, (images, labels) in enumerate(test_loader):
        print("batch", batch_idx)
        batch_size = images.shape[0]
        n_images += batch_size
        images, labels = images.to(device), labels.to(device)
        images_h, _ = attack.perturb(images, labels, debug=True)
        with torch.no_grad():
            preds = model(images_h).argmax(-1)
            count_correct += preds.eq(labels).sum()
            if log_preds:
                batch_df = pd.DataFrame({"attack" : ["adv"] * batch_size, "target" : [None] * batch_size, "gt" : labels.cpu(), "prediction" : preds.cpu()})
                preds_df = pd.concat([preds_df, batch_df])
    adv_acc = float(count_correct) / float(n_images)
    print(f"adv\t{adv_acc}")
    if path is not None:
        print("saving adv accuracy to", path)
        with open(path, "w") as fp:
            fp.write(str(adv_acc)) 
    preds_summ_df = preds_df.groupby(preds_df.columns.to_list(), as_index=False, dropna=False).size()
    return adv_acc, preds_summ_df


def evaluate_clean(model, test_loader, n_classes, path=None, log_preds=False):
    model.eval()
    class_correct = torch.zeros(n_classes).type(dtype=torch.float)
    class_n_images = torch.zeros(n_classes).type(dtype=torch.float)
    total_correct = 0
    preds_df = pd.DataFrame()
    for batch_idx, (images, labels) in enumerate(test_loader):
        print("batch", batch_idx)
        batch_size = len(labels)
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(images).argmax(-1)
            for source in range(n_classes):
                src_labels = labels[labels == source].to(device)
                print(f'source: {source} num imgs: {len(src_labels)}')
                if len(src_labels) == 0: continue
                class_n_images[source] += len(src_labels)
                src_preds = preds[labels == source]
                num_correct = src_preds.eq(src_labels.long()).sum()
                class_correct[source] += num_correct.item()
                total_correct += num_correct.item()
                print(f'correctly classified: {num_correct.item()}')
            if log_preds:
                batch_df = pd.DataFrame({"attack" : ["clean"] * batch_size, "target" : [None] * batch_size, "gt" : labels.cpu(), "prediction" : preds.cpu()})
                preds_df = pd.concat([preds_df, batch_df])
    n_images = class_n_images.sum().item()
    class_acc = torch.div(class_correct, class_n_images)
    total_acc = float(total_correct) / float(n_images)
    for source in range(n_classes):
        print(f'source {source} num_images: {class_n_images[source]} accuracy: {class_acc[source]}')
    print(f'total_acc = {total_correct} / {n_images} = {total_acc}')
    if path:
        df = pd.DataFrame(class_acc.numpy())  # convert to a dataframe
        df.loc['mean'] = total_acc
        print("saving results to", path)
        df.to_csv(path, index=False)  # save to file
    preds_summ_df = preds_df.groupby(preds_df.columns.to_list(), as_index=False, dropna=False).size()
    return total_acc, class_acc, preds_summ_df


def evaluate_attack(model, attack, test_loader, n_classes, path=None, log_preds=False):
    model.eval()
    correct_mat = torch.zeros((n_classes, n_classes)).type(dtype=torch.float)
    n_images = torch.zeros((n_classes, 1))
    preds_df = pd.DataFrame()
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f'batch: {batch_idx}')
        batch_size = len(labels)
        images = images.to(device)
        labels = labels.to(device)
        for target in range(n_classes):
            targets = torch.ones(images.shape[0]).to(device) * target
            images_h = attack.perturb(images.to(device), targets.long(), labels)[0].detach()
            with torch.no_grad():
                preds = model(images_h).argmax(-1)
                for source in range(n_classes):
                    src_labels = labels[labels == source].to(device)
                    if len(src_labels) == 0: continue
                    src_preds = preds[labels == source]
                    num_correct = src_preds.eq(src_labels.long()).sum()
                    correct_mat[source][target] += num_correct.item()
                if log_preds:
                    target_df = pd.DataFrame({"attack" : ["targeted"] * batch_size, "target" : [target] * batch_size, "gt" : labels.cpu(), "prediction" : preds.cpu()})
                    target_df = target_df.groupby(["attack", "target", "gt", "prediction"], as_index=False, dropna=False).size()
                    preds_df = pd.concat([preds_df, target_df])
                    preds_df = preds_df.groupby(["attack", "target", "gt", "prediction"], as_index=False, dropna=False)["size"].sum()
        for source in range(n_classes):
            n_images[source][0] += len(labels[labels == source])
    for source in range(n_classes):
        for target in range(n_classes):
            print(f'acc {source}->{target}: {correct_mat[source][target]} / {n_images[source][0]} = {correct_mat[source][target] / float(n_images[source][0])}')
    acc_mat = torch.div(correct_mat, n_images)
    if path:
        df = pd.DataFrame(acc_mat.numpy())  # convert to a dataframe
        print("saving results to", path)
        df.to_csv(path, index=False)  # save to file
    return acc_mat, preds_df


def evaluate_multi_targets_attack(model, attack, test_loader, n_classes, targets_mat, path=None, log_preds=False):
    model.eval()
    
    targets_mask = (targets_mat.sum(1) > 0) # find classes with at least 1 target
    no_targets_mask = (targets_mat.sum(1) == 0) # find classes with 0 targets
    clean_classes = no_targets_mask.nonzero().squeeze(1)
    print(f"clean_classes={clean_classes}")
    multi_target_classes = targets_mask.nonzero().squeeze(1)
    print(f"multi_target_classes={multi_target_classes}")
            
    correct_arr = torch.zeros(n_classes).type(dtype=torch.float)
    n_images = torch.zeros(n_classes)
    preds_df = pd.DataFrame()
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f'batch: {batch_idx}')
        batch_size = len(labels)
        images = images.to(device)
        labels = labels.to(device)
        
        # attack images of classes with multi targets (at least one target)
        mt_mask = sum(labels==i for i in multi_target_classes).bool()    
        mt_labels = labels[mt_mask]
        mt_images = images[mt_mask]
        strategic_multi_labels = translate_multi_labels_by_strategy(labels=mt_labels, strat=targets_mat)
        mt_images_h = attack.perturb(mt_images, mt_labels, multi_targets_mask=strategic_multi_labels, debug=True)[0]
        images_h = images
        images_h[mt_mask] = mt_images_h
        with torch.no_grad():
            preds = model(images_h).argmax(-1)
            for source in range(n_classes):
                src_labels = labels[labels == source].to(device)
                if len(src_labels) == 0: continue
                src_preds = preds[labels == source]
                num_correct = src_preds.eq(src_labels.long()).sum()
                correct_arr[source] += num_correct.item()
            if log_preds:
                batch_df = pd.DataFrame({"attack" : ["multi_targets_" + str(attack.loss_fn)] * batch_size, "target" : ["multi"] * batch_size, "gt" : labels.cpu(), "prediction" : preds.cpu()})
                batch_df = batch_df.groupby(["attack", "target", "gt", "prediction"], as_index=False, dropna=False).size()
                preds_df = pd.concat([preds_df, batch_df])
                preds_df = preds_df.groupby(["attack", "target", "gt", "prediction"], as_index=False, dropna=False)["size"].sum()
        for source in range(n_classes):
            n_images[source] += len(labels[labels == source])
    for source in range(n_classes):
        print(f'acc source={source}: {correct_arr[source]} / {n_images[source]} = {correct_arr[source] / float(n_images[source])}')
    acc_arr = torch.div(correct_arr, n_images)
    total_acc = float(correct_arr.sum()) / float(n_images.sum())
    if path:
        df = pd.DataFrame(acc_arr.numpy())  # convert to a dataframe
        df.loc['mean'] = total_acc
        print("saving results to", path)
        df.to_csv(path, index=False)  # save to file
    return acc_arr, preds_df
    

# run
def main():
    ts = datetime.now().strftime("%d%m%y_%H%M%S")
    args = parse_args()
    print(f'Chosen args : {args}')
    # Set seed for reproducibility
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_name = 'GTSRB' if args.dataset == 'bazyl/GTSRB' else args.dataset # alter bazyl/GTSRB to GTSRB
    model_path = args.model_path
    
    preds_df = pd.DataFrame(columns=["attack", "target", "gt", "prediction"]) if args.log_preds else None

    # params
    eps_iter_val = (2.5 * args.eps / args.nb_iter) * args.eps_iter_factor
    params = {"eps": args.eps,
              "nb_iter": args.nb_iter,
              "eps_iter": eps_iter_val,
              "ord": np.inf if args.ord == 0 else 2}

    print(f'arch: {args.arch}\ndataset: {ds_name}\nuse_aug: {args.use_aug}\ntest_loss: {args.test_loss_list}\nmodel path: {model_path}')

    # load pre-trained model
    test_model = load_model(args.arch, args.n_classes, ds_name, model_path)
    test_model.eval()

    # get data
    image_size = (32, 32)
    batch_size = 64 if image_size[0] > 32 else 500
    data_loader, class_counts = get_data_loader(ds_name, args.set, batch_size=batch_size, image_size=image_size, class_list=args.class_list)
    assert len(class_counts) == args.n_classes
    counts_path = f'{args.results_dir}/class_counts.csv'
    counts_df = pd.DataFrame(class_counts.numpy())  # convert to a dataframe
    print("saving class counts to", counts_path)
    counts_df.to_csv(counts_path, index=False)  # save to file

    if args.targets_mat is not None:
        print(f"evaluating against multi targets attacker")
        for mt_loss in args.mt_loss_list:
            multi_targets_attack = create_attck(test_model, is_targeted=True, loss_num=mt_loss, params=params)
            eval_path = f'{args.results_dir}/eval_multi_targets_{args.arch}_{ds_name}_aug_{args.use_aug}_{mt_loss}.csv'
            acc, multi_preds_df = evaluate_multi_targets_attack(test_model, multi_targets_attack, data_loader, args.n_classes, args.targets_mat, path=eval_path, log_preds=args.log_preds)
            preds_df = pd.concat([preds_df, multi_preds_df])
        
    if not args.omit_adv:
        # measure adversarial accuracy
        adv_attack = create_attck(test_model, is_targeted=False, loss_num=0, params=params)
        eval_vs_adv_path = f'{args.results_dir}/eval_vs_adv.txt'
        adv_acc, adv_preds_df = evaluate_adv(test_model, adv_attack, data_loader, path=eval_vs_adv_path, log_preds=args.log_preds)
        preds_df = pd.concat([preds_df, adv_preds_df])

    if not args.omit_clean:
        # measure clean accuracy
        eval_clean_path = f'{args.results_dir}/eval_clean_mat_{args.arch}_{ds_name}_aug_{args.use_aug}_{ts}.csv'
        _, _, clean_preds_df = evaluate_clean(test_model, data_loader, args.n_classes, path=eval_clean_path, log_preds=args.log_preds)
        preds_df = pd.concat([preds_df, clean_preds_df])
    
    # measure targeted attacks accuracies *per class*
    for test_loss in args.test_loss_list:
        print(f"evaluating against loss {test_loss}")
        targeted_attack = create_attck(test_model, is_targeted=True, loss_num=test_loss, params=params)
        eval_path = f'{args.results_dir}/eval_all_mat_{args.arch}_{ds_name}_aug_{args.use_aug}_test_loss_{test_loss}_{ts}.csv'
        acc, targeted_preds_df = evaluate_attack(test_model, targeted_attack, data_loader, args.n_classes, path=eval_path, log_preds=args.log_preds)
        preds_df = pd.concat([preds_df, targeted_preds_df])
        for target in range(args.n_classes):
            target_acc = acc[target][:].sum() #- acc[target][target]
            print("target:", target, "acc: ", target_acc)
        

    if args.log_preds:
        preds_path = f'{args.results_dir}/preds.csv'
        print("saving predictions to", preds_path)
        preds_df.to_csv(preds_path, index=False)

if __name__ == '__main__':
    main()
