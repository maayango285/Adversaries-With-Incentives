import torch
from models.vgg import *
from models.model_zoo import PreActResNet, WideResNet
from models.ViT import ViT
from attacks.attacks import LinfPGDAttack, L2PGDAttack
import argparse
from tqdm import tqdm
import os
from data_utils import get_dataset
import numpy as np
import random
from utils import get_random_strategy, get_col_strategy, training_strategy_dict, translate_label_by_strategy, translate_multi_labels_by_strategy, get_random_subset_strategy
from torchvision import transforms
import platform # check running platform
from datetime import datetime
import pickle
from accelerate import Accelerator

# Global variables
accelerator = Accelerator()
device = accelerator.device
arch_dict = {
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    'vgg19': vgg19,
    'resnet': PreActResNet,
    'wrn': WideResNet,
    'ViT' : ViT
}

GTSRB_2g_class_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

def parse_args():
    parser = argparse.ArgumentParser(description='Strategic Adversarial Training')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--output_dir', type=str, default='output_dir', help='path for saving checkpoints')
    # Optimization HPs
    parser.add_argument('--bs', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # Attack HPs
    parser.add_argument('--attack', type=int, default=0, help="0 for Linf, 1 for L2")
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--nb_iter', type=int, default=7)
    # Training strategy
    parser.add_argument('--training_strategy', type=int, default=8)
    parser.add_argument('--gt_strat', nargs='*', type=int, default=None, help="pass ground truth srategy as list - used by oracle strategies")
    parser.add_argument('--targets_mat_path', type=str, default=None, help="pass path to a pickled binary mat describing each source's targets subset. only applicable to specific strategies")
    parser.add_argument('--flip_prob', type=float, default=0.1, help="probability for using random target - used by noisy oracles and multi_targets")
    parser.add_argument("--use_aug", type=int, default=1, help="Use data augmentations?")
    # Testing strategy
    parser.add_argument('--testing_strategy', type=str, default='random')
    # Data HPs
    parser.add_argument('--dataset', type=str, default='bazyl/GTSRB')
    parser.add_argument('--class_list', nargs='*', type=int, default=None, help="pass partial class list")
    # Model HPs
    parser.add_argument('--arch', type=str, default='vgg11_bn')
    parser.add_argument('--initial_checkpoint', type=str, default=None, help='path to initial model parameters')
    # Attack parametes
    parser.add_argument('--attack_loss', type=int, default=0, help="0: adv - CE on source | 8: multi-targeted - CE on max target")
    # Performance
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    args, _ = parser.parse_known_args()

    if args.dataset == 'bazyl/GTSRB':
        args.dataset = 'GTSRB'  # alter bazyl/GTSRB to GTSRB for consistency

    # additional arguments
    args.image_size = (32, 32)
    if args.dataset == 'GTSRB_2g':
        args.class_list = GTSRB_2g_class_list
    args.n_classes = -1
    if args.class_list is not None:
        args.n_classes = len(args.class_list)
    elif args.dataset == 'GTSRB':
        args.n_classes = 43
    else:
        args.n_classes = 10

    args.gpu_ids = None if device == 'cpu' else [i for i in range(args.num_gpus)]
    print(args.gpu_ids)
    args.targets_mat = None
    if args.targets_mat_path is not None:
        args.targets_mat = pickle.load(open(args.targets_mat_path, 'rb'))
    else: # all targets are allowed, except self
        args.targets_mat = torch.ones((args.n_classes, args.n_classes)).fill_diagonal_(0)
    return args


def load_model(arch, num_classes, ds_name, initial_checkpoint):
    model = arch_dict[arch](num_classes, ds_name).to(device)
    if initial_checkpoint is not None:
        try:
            m, u = model.load_state_dict(torch.load(initial_checkpoint))
            assert len(m) == 0 and len(u) == 0
        except RuntimeError:
            print("Encountered RuntimeError while loading state_dict. Trying to remove module from all keys.")
            orig_state_dict = torch.load(initial_checkpoint)
            state_dict = {key.replace("module.", ""): value for key, value in orig_state_dict.items()}
            m, u = model.load_state_dict(state_dict)
            assert len(m) == 0 and len(u) == 0
    return model
        

def aug(x, image_size=32):
    trans = transforms.Compose([transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip()])
    return trans(x)


def adjust_learning_rate(optimizer, epoch, config):
    """a multistep learning rate drop mechanism"""
    lr = config["lr"]
    epochs = config["epochs"]
    if epoch >= 0.5 * epochs:
        lr = config["lr"] * 0.1
    if epoch >= 0.75 * epochs:
        lr = config["lr"] * 0.01
    if epoch >= epochs:
        lr = config["lr"] * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    accelerator.print(f'epoch : {epoch} -> lr : {lr} time : {datetime.now().strftime("%H:%M:%S")}')


# Train function
def train_epoch(f, loader, optimizer, train_strat, attack, gt_strat, args):
    '''
    Train for an epoch
    :param f: a feed forward neural network
    :param loader: a train loader
    :return:
    '''
    clean_acc = 0
    adv_acc = 0
    adv_total_loss = 0
    n_images = 0

    rt_attack = None
    if train_strat == 'multi_targets' and args.flip_prob > 0:
        rt_attack = LinfPGDAttack if args.attack == 0 else L2PGDAttack
        rt_attack = rt_attack(predict=f, eps=args.eps, nb_iter=args.nb_iter, eps_iter=2.5 * args.eps / args.nb_iter,
                                targeted=True, loss_fn=0)
    for batch_idx, (images, labels) in enumerate(loader):    
        print(f"batch: {batch_idx} size: {len(labels)}")
        optimizer.zero_grad()
        n_images += images.shape[0]
        if args.use_aug > 0:
            images = aug(images, args.image_size)
        images, labels = images.to(device), labels.to(device)
        images_h = None
        # Act according to training strategy
        # 0: 'vanilla',
        # 1: 'worst_case',
        # 2: 'oracle',
        # 3: 'noisy_oracle',
        # 4: 'multi_targets'
        # 5: 'noisy_subset_oracle'
        labels_h = labels.clone()
        if train_strat == 'vanilla':
            
            images_h = images
            
        elif train_strat == 'worst_case':
            
            images_h, delta = attack.perturb(images, labels)
            
        elif train_strat == 'multi_targets':
            
            curr_targets_mat = args.targets_mat.clone().detach()
            # choose classes at flip probability to have a random single target
            flips = torch.rand(size=(args.n_classes,)) >= 1 - args.flip_prob
            print(f"flips={flips}")
            rand_target_classes = flips.nonzero().squeeze(1)
            print(f"rand_target_classes={rand_target_classes}")
            
            # update random targets in curr_targets_mat
            rand_strat = get_random_strategy(args.n_classes)
            print(f'rand_strat={rand_strat}')
            for c in rand_target_classes:
                c = c.item()
                curr_targets_mat[c] = False
                rand_target, budget = rand_strat[c]
                curr_targets_mat[c][rand_target] = True

            # extract classes with no targets (clean)
            clean_mask = (curr_targets_mat.sum(1) == 0)
            clean_classes = clean_mask.nonzero().squeeze(1)
            print(f"clean_classes={clean_classes}")
            
            # extract classes with multi targets
            multi_targets_mask = (curr_targets_mat.sum(1) != 0)
            multi_target_classes = multi_targets_mask.nonzero().squeeze(1)
            print(f"multi_target_classes={multi_target_classes}")

            # attack images of classes with multi targets (not clean)
            mt_mask = sum(labels==i for i in multi_target_classes).bool()    
            mt_labels = labels[mt_mask]
            mt_images = images[mt_mask]
            strategic_multi_labels = translate_multi_labels_by_strategy(labels=mt_labels, strat=curr_targets_mat)
            mt_images_h = attack.perturb(mt_images, mt_labels, multi_targets_mask=strategic_multi_labels, debug=True)[0]
            
            images_h = images
            images_h[mt_mask] = mt_images_h
            
        elif train_strat == 'oracle':
            strategic_labels, budget = translate_label_by_strategy(labels=labels, strat=gt_strat)
            images_h, delta = attack.perturb(images, strategic_labels, labels, budget)

        elif train_strat == 'noisy_oracle':
            not_flips = torch.rand(size=(args.n_classes,)) < 1 - args.flip_prob
            noisy_oracle = get_random_strategy(args.n_classes)
            for j, not_flip in enumerate(not_flips):
                if not_flip:
                    noisy_oracle[j] = gt_strat[j]
            print(f"noisy strat: {noisy_oracle}")
            strategic_labels, budget = translate_label_by_strategy(labels=labels, strat=noisy_oracle)
            images_h = attack.perturb(images, strategic_labels, labels, budget)[0]
            
        elif train_strat == 'noisy_subset_oracle':
            # the first 0.1 portion of args.flip_prob is distributed across all classes
            fp_all_threshold = args.flip_prob if args.flip_prob <= 0.1 else 0.1
            # the rest distributes across the subset specified in args.targets_mat
            fp_subset_threshold = args.flip_prob if args.flip_prob > 0.1 else 0
            print(f'gt strat: {gt_strat}')
            subset_rand_strat = get_random_subset_strategy(args.n_classes, args.targets_mat)
            print(f'rand subset strat: {subset_rand_strat}')
            rand_strat = get_random_strategy(args.n_classes)
            print(f'rand strat: {rand_strat}')
            noisy_oracle = gt_strat.copy() 
            flips = torch.rand(size=(args.n_classes,))
            print(f'flips: {flips}')
            # rand target: [0, min(args.flip_prob, 0.1)) | rand subset target: [min(args.flip_prob, 0.1), args.flip_prob) | gt: [args.flip_prob, 1)
            for source in range(args.n_classes):
                if flips[source] < fp_all_threshold:
                    noisy_oracle[source] = rand_strat[source]
                elif flips[source] < fp_subset_threshold:
                    noisy_oracle[source] = subset_rand_strat[source]
            print(f"flipped strat={noisy_oracle}")
            strategic_labels, budget = translate_label_by_strategy(labels=labels, strat=noisy_oracle)
            images_h = attack.perturb(images, strategic_labels, labels, budget)[0]
        
        else:
            raise Exception('invalid training strategy')

        # step
        f.train()
        optimizer.zero_grad()
        preds = f(images_h)
        loss = torch.nn.CrossEntropyLoss()(preds, labels_h)
            
        accelerator.backward(loss)
        optimizer.step()

        with torch.no_grad():
            clean_acc += f(images).argmax(-1).eq(labels).sum()
            adv_acc += (f(images_h).argmax(-1).eq(labels_h).sum() / (labels_h.shape[0] / (args.bs / accelerator.num_processes)))
            adv_total_loss += loss
        
    return clean_acc / n_images, adv_acc / n_images, adv_total_loss


# Eval function
def test_epoch(f, loader, optimizer, attack, gt_strat, args): #TODO: can we lose the optimizer here?
    f.eval()
    clean_acc = 0
    adv_acc = 0
    n_images = 0
    tstrat = training_strategy_dict[args.training_strategy]
    for batch_idx, (images, labels) in enumerate(loader):
        n_images += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        images_h = None
        if tstrat == 'multi_targets':
            strategic_multi_labels = translate_multi_labels_by_strategy(labels=labels, strat=args.targets_mat)
            images_h = attack.perturb(images, labels, multi_targets_mask=strategic_multi_labels)[0]
        else:
            strategic_labels, budget = translate_label_by_strategy(labels=labels, strat=gt_strat)
            images_h, _ = attack.perturb(images, strategic_labels, labels, budget)
        optimizer.zero_grad()
        with torch.no_grad():
            clean_acc += f(images).argmax(-1).eq(labels).sum()
            adv_acc += f(images_h).argmax(-1).eq(labels).sum()
    return clean_acc / n_images, adv_acc / n_images


def process(args):
    # Set seed for reproducability
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Output dir for saving checkpoints
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    strategy_symbol = str(args.training_strategy)
    
    gt_strat = None
    if args.gt_strat is None:
        # Get random ground-truth strategy
        gt_strat = get_random_strategy(args.n_classes)
    else:
        # Parse user defined gt strategy
        if len(args.gt_strat) != args.n_classes:
            raise Exception('invalid gt strategy')
        gt_strat = {i : (args.gt_strat[i], 1) for i in range(args.n_classes)}
        strategy_symbol += "_" + "".join([str(c) for c in args.gt_strat])
    print(f'gt_strat:\n{gt_strat}')

    model_name = f'{args.arch}_{args.dataset}_tstrat_{strategy_symbol}_seed_{args.seed}_aug_{args.use_aug}_attack_loss_{args.attack_loss}.pt'
    save_path = f'{args.output_dir}/' + model_name + '.pt'
    
    # Get training strategy as a string
    train_strat = training_strategy_dict[args.training_strategy]
    print(f'train_strat = {train_strat}')
    
    config = vars(args)

    # Get data (GTSRB | GTSRB_2g | cifar10 | cifar100)
    ds_to_load = args.dataset
    if ds_to_load in ['GTSRB', 'GTSRB_2g']:
        ds_to_load = 'bazyl/GTSRB'
    train_set, test_set = get_dataset(data=ds_to_load, image_size=args.image_size, class_list=args.class_list)
    batch_size = int(args.bs / accelerator.num_processes)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=2)
    test_loader = accelerator.prepare(test_loader)

    # Get model and optimizer
    orig_model = load_model(args.arch, args.n_classes, args.dataset, args.initial_checkpoint)
    optimizer = torch.optim.SGD(orig_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    can_compile = train_strat != 'multi_targets' # may use variable batch_size
    model = torch.compile(orig_model) if can_compile else orig_model
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # Train attack
    attack = None
    assert (train_strat != 'worst_case') or (args.attack_loss == 0)
    if train_strat != 'vanilla':  # Attack is used for all the non vanilla cases
        attack = LinfPGDAttack if args.attack == 0 else L2PGDAttack
        attack = attack(predict=model, eps=args.eps, nb_iter=args.nb_iter, eps_iter=2.5 * args.eps / args.nb_iter,
                        targeted=True if train_strat not in ['worst_case', 'worst_case_subset'] else False,
                        loss_fn=args.attack_loss)
    # Test attack
    attack_test = LinfPGDAttack if args.attack == 0 else L2PGDAttack
    attack_test = attack_test(predict=model, eps=args.eps, nb_iter=20, eps_iter=2.5 * args.eps / 20,
            targeted=True if train_strat != 'worst_case' else False,
            loss_fn=args.attack_loss)
    
    # Runner
    torch.save(model.state_dict(), save_path)
    for epoch in tqdm(range(args.epochs)):
        # Adjust learning rate as needed
        adjust_learning_rate(optimizer, epoch, config)
        
        # Train epoch
        clean_acc_train, adv_acc_train, adv_loss = train_epoch(f=model, loader=train_loader, optimizer=optimizer,
                                                     train_strat=train_strat, attack=attack, gt_strat=gt_strat, args=args)
        clean_acc_train = accelerator.reduce(clean_acc_train, reduction="mean")
        adv_acc_train = accelerator.reduce(adv_acc_train, reduction="mean")
        adv_loss = accelerator.reduce(adv_loss, reduction="mean")
        accelerator.print(f'Train clean: {clean_acc_train}, Train adv: {adv_acc_train} Loss: {adv_loss}')
        
        # Test epoch
        clean_acc_test, adv_acc_test = test_epoch(f=model, loader=test_loader, optimizer=optimizer,
                                                  attack=attack_test, gt_strat=gt_strat, args=args)
        clean_acc_test = accelerator.reduce(clean_acc_test, reduction="mean")
        adv_acc_test = accelerator.reduce(adv_acc_test, reduction="mean")
        accelerator.print(f'Test clean: {clean_acc_test}, Test adv: {adv_acc_test}')
        # Save current state
        torch.save(orig_model.state_dict(), save_path)
        
    if accelerator.is_main_process:
        print("saving final model...")
        torch.save(orig_model.state_dict(), save_path)
    print("DONE")


def main():
    args = parse_args()
    accelerator.print(f'Chosen args : {args}')
    accelerator.print(f'torch.cuda.device_count() : {torch.cuda.device_count()}')
    torch.set_printoptions(threshold= args.n_classes * args.n_classes)
    process(args)


if __name__ == '__main__':
    main()
