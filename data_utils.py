from datasets import load_dataset
import torchvision
import torch
from torch.utils.data import TensorDataset


# Convert datasets object into torch dataset
def ists2tensor(img_list, lbl_list, size=(32, 32)):
    resize = torchvision.transforms.Resize(size)
    data_f, labels_f = [], []
    for i in range(len(img_list)):
        if len(img_list[i].size()) == 3:
            im = img_list[i]
            is_CHW = im.size()[0] == 3
            if not is_CHW:
                im = im.permute(2, 0, 1)
            d = resize((im.type(torch.FloatTensor) / 255.).unsqueeze(0))  # in [0,1]
            data_f.append(d)
            labels_f.append(lbl_list[i])

    data = torch.cat(data_f)
    labels = torch.tensor(labels_f)
    dataset = TensorDataset(data, labels)
    return dataset


def extract_classes(imgs, lbls, class_list):
    print(f'len(imgs)={len(imgs)}')
    print(f'len(lbls)={len(lbls)}')
    new_imgs = []
    new_lbls = []
    for i in range(len(lbls)):
        if lbls[i] in class_list:
            new_imgs.append(imgs[i])
            new_lbls.append(lbls[i]) 
    print(f'len(new_imgs)={len(new_imgs)}')
    print(f'len(new_lbls)={len(new_lbls)}')
    return new_imgs, new_lbls

def get_dataset(data='GTSRB', image_size=(32, 32), class_list=None):
    supported = ['bazyl/GTSRB', 'cifar10', 'cifar100']
    if data not in supported:
        raise NotImplementedError(f'Currently we support only {supported}')
    train_imgs, train_lbls = None, None
    test_imgs, test_lbls = None, None
    
    if data in ['bazyl/GTSRB', 'cifar10', 'cifar100']:
        dataset = load_dataset(data)
        dataset = dataset.with_format("torch")
        train_data, test_data = dataset['train'], dataset['test']
        if data == 'cifar10':
            train_imgs, train_lbls = train_data['img'], train_data['label']
            test_imgs, test_lbls = test_data['img'], test_data['label']
        elif data == 'cifar100':
            train_imgs, train_lbls = train_data['img'], train_data['fine_label']
            test_imgs, test_lbls = test_data['img'], test_data['fine_label']
        else:
            train_imgs, train_lbls = train_data['Path'], train_data['ClassId']
            test_imgs, test_lbls = test_data['Path'], test_data['ClassId']
    
    if class_list is not None:
        print(f'class_list={class_list}')
        test_imgs, test_lbls = extract_classes(test_imgs, test_lbls, class_list)
        train_imgs, train_lbls = extract_classes(train_imgs, train_lbls, class_list)
        
        # re-number labels from 0
        test_lbls = torch.tensor(test_lbls)
        train_lbls = torch.tensor(train_lbls)
        for i, c in enumerate(class_list):
            test_lbls[test_lbls == c] = i
            train_lbls[train_lbls == c] = i
    
    print(f"image size: {image_size}")
    train_set = ists2tensor(train_imgs, train_lbls, size=image_size)
    test_set = ists2tensor(test_imgs, test_lbls, size=image_size)
    
    return train_set, test_set

