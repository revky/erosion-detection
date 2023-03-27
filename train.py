import math
from pathlib import Path

import albumentations as A
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.utils import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score, binary_recall, \
    binary_jaccard_index, binary_precision
from tqdm.auto import tqdm

from dataset import SegmentDataset


def get_mean_std(args):
    p = Path(args['mean_std_path'])
    if p.exists():
        file = torch.load(p)
        print('Mean and std loaded from cashe.')
        return file['mean'], file['std']

    ds = SegmentDataset(image_path=args['image_path'],
                        mask_path=args['mask_path'],
                        patch_size=args['patch_size'],
                        transform=ToTensorV2())

    dl = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=args['num_workers'])
    images, _ = next(iter(dl))
    print(images.shape)
    images = images / 255
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    torch.save({'mean': mean, 'std': std}, args['mean_std_path'])
    return mean, std


def get_transform(train, args):
    trans = []
    if train:
        trans.extend([

        ])

    trans.extend([
        A.Normalize(*get_mean_std(args)),
        ToTensorV2(transpose_mask=True)
    ])
    trans = A.Compose(trans)
    return trans


def get_samplers(train_len, args):
    indices = torch.randperm(train_len)

    # Count limit number for train and val ids
    split = int(math.ceil(args['test_ratio'] * train_len))
    train_idx, test_idx = indices[split:], indices[:split]

    # Create sampler for given ids
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    return train_sampler, test_sampler


def get_datasets(args):
    train_dataset = SegmentDataset(image_path=args['image_path'],
                                   mask_path=args['mask_path'],
                                   patch_size=args['patch_size'],
                                   transform=get_transform(True, args),
                                   path_to_save=args['path_to_save'],
                                   save_images=args['save_images'])

    test_dataset = SegmentDataset(image_path=args['image_path'],
                                  mask_path=args['mask_path'],
                                  patch_size=args['patch_size'],
                                  transform=get_transform(False, args),
                                  path_to_save=args['path_to_save'],
                                  save_images=args['save_images'])

    return train_dataset, test_dataset


def train_epoch(model, loss_fn, optimizer, train_loader, epoch, device, writer):
    model.train()
    train_loss = 0.0
    for (images, targets) in tqdm(train_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    writer.add_scalar('Train/loss', train_loss, epoch)
    return train_loss


def test_epoch(model, loss_fn, test_loader, metrics, epoch, device, writer):
    model.eval()
    metrics_sum = {}
    test_loss = 0.0
    with torch.inference_mode():
        for images, targets in tqdm(test_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

            for name, fn in metrics.items():
                metrics_sum.setdefault(name, 0.0)
                metrics_sum[name] += fn(outputs, targets)

    metrics_mean = {key: value / len(test_loader) for key, value in metrics_sum.items()}

    for key, value in metrics_mean.items():
        writer.add_scalar(f'Test/{key}', value, epoch)

    test_loss /= len(test_loader)
    writer.add_scalar('Test/loss', test_loss, epoch)
    return test_loss, metrics_mean


def main(args):
    device = torch.device(args['device'])
    train_dataset, test_dataset = get_datasets(args)
    train_len = len(train_dataset)
    train_sampler, test_sampler = get_samplers(train_len, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        sampler=train_sampler,
        num_workers=args['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        sampler=test_sampler,
        num_workers=args['num_workers']
    )

    model = smp.Unet(
        encoder_name=args['encoder_name'],
        encoder_weights=args['encoder_weights']
    )
    model.to(device)

    params_to_optimize = [
        {'params': [p for p in model.decoder.parameters() if p.requires_grad]},
        {'params': [p for p in model.encoder.parameters() if p.requires_grad]}
    ]

    # TODO
    # Move to function
    # Add parameters to optimizer if performance sucks
    # Same with lr scheduler
    optimizer = torch.optim.SGD(params_to_optimize, lr=args['lr'])

    # TODO
    # Move to function
    # Same with lr scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, cooldown=2, min_lr=1e-7)

    # TODO
    # Move to function
    loss_fn = losses.BCEWithLogitsLoss() + losses.DiceLoss()

    # TODO
    # Move to function
    metrics = {
        'precision': binary_precision,
        'recall': binary_recall,
        'accuracy': binary_accuracy,
        'f1': binary_f1_score,
        'iou': binary_jaccard_index,
    }

    # TODO
    # Add resuming
    best_test_loss = float('inf')

    writer = SummaryWriter(log_dir='./runs')
    for epoch in range(1, args['epochs']):
        train_loss = train_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            epoch,
            device,
            writer
        )
        test_loss, eval_metrics = test_epoch(
            model,
            loss_fn,
            test_loader,
            metrics,
            epoch,
            device,
            writer
        )

        lr_scheduler.step(test_loss)
        header = f'EPOCH {epoch}|'
        train_loss_summary = f'TRAIN LOSS: {train_loss:.3f}| '
        test_loss_summary = f'TEST LOSS: {test_loss:.3f}| '
        eval_summary = ''.join(f'TEST_{key.upper()}: {value:.3f}| '.format(key) for key, value in eval_metrics.items())

        print(header, train_loss_summary, test_loss_summary, eval_summary)
