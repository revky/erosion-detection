import argparse
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


def get_transform(train, args):
    trans = []
    if train:
        trans.extend([
            A.OneOf([
                A.RandomSizedCrop(min_max_height=(100, 151),
                                  height=args.patch_size,
                                  width=args.patch_size, p=0.5),
                A.PadIfNeeded(min_height=args.patch_size,
                              min_width=args.patch_size, p=0.5)
            ], p=1),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=0.8)])
    trans.extend([
        ToTensorV2(transpose_mask=True)
    ])
    trans = A.Compose(trans)
    return trans


def get_samplers(train_len, args):
    indices = torch.randperm(train_len)
    # Count limit number for train and val ids
    split = int(math.ceil(args.test_ratio * train_len))
    train_idx, test_idx = indices[split:], indices[:split]
    # Create sampler for given ids
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    return train_sampler, test_sampler


def get_datasets(args):
    images_dir = Path(args.images_dir) / str(args.patch_size)
    masks_dir = Path(args.masks_dir) / str(args.patch_size)

    train_dataset = SegmentDataset(images_dir=images_dir,
                                   masks_dir=masks_dir,
                                   transform=get_transform(True, args),
                                   add_empty_masks=args.add_masks)
    # TODO
    # SUS (2x generated masks idx, may cause problems)
    test_dataset = SegmentDataset(images_dir=images_dir,
                                  masks_dir=masks_dir,
                                  transform=get_transform(False, args),
                                  add_empty_masks=args.add_masks)

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


def get_model(args):
    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights
    )
    return model


def get_optimizer(model, args):
    if args.opt == 'adam':
        return torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    raise NotImplementedError


def get_scheduler(optimizer, args):
    if args.scheduler == 'plat':
        return ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.5, cooldown=2, min_lr=1e-7)
    raise NotImplementedError


def get_loss_fn(args):
    if args.loss_fn == 'bce':
        return losses.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.class_weight]).to(args.device))

    if args.loss_fn == 'dice':
        return losses.DiceLoss()

    if args.loss_fn == 'focal':
        return smp.losses.FocalLoss(mode='binary', alpha=torch.Tensor([args.class_weight]).to(args.device))
    raise NotImplementedError


def main(args):
    device = torch.device(args.device)
    train_dataset, test_dataset = get_datasets(args)
    train_len = len(train_dataset)
    train_sampler, test_sampler = get_samplers(train_len, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers
    )

    model = get_model(args).to(device)
    loss_fn = get_loss_fn(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_scheduler(optimizer, args)

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
    best_precision = 0.0
    writer = SummaryWriter(log_dir='./runs')

    for epoch in range(1, args.epochs + 1):
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
        lr_scheduler.step(eval_metrics['precision'])

        header = f'EPOCH {epoch}|'
        train_loss_summary = f'TRAIN LOSS: {train_loss:.3f}| '
        test_loss_summary = f'TEST LOSS: {test_loss:.3f}| '
        eval_summary = ''.join(f'TEST_{key.upper()}: {value:.3f}| '.format(key) for key, value in eval_metrics.items())
        print(header, train_loss_summary, test_loss_summary, eval_summary)

        if eval_metrics['precision'] > best_precision:
            best_precision = eval_metrics['precision']
            torch.save(model.state_dict(), args.best_model_path)
            print('Saving new best model at', args.best_model_path)


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--patch_size", default=256, type=int, help="desired patch size")
    parser.add_argument("--images_dir", default="data/images", type=str, help="path to save new images")
    parser.add_argument("--masks_dir", default="data/masks", type=str, help="path to save new masks")
    parser.add_argument("--test_ratio", default=0.2, type=float, help="train/test ratio")
    parser.add_argument("--device", default='cuda', type=str, help="device type")
    parser.add_argument("--batch_size", default=4, type=int, help="pipeline batch size")
    parser.add_argument("--num_workers", default=0, type=int, help="cpu workers")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=2e-5, type=float, help="optimizer weight decay")
    parser.add_argument("--epochs", default=250, type=int, help="number of epochs")
    parser.add_argument("--encoder_name", default='efficientnet-b2', type=str, help="pretrained model architecture")
    parser.add_argument("--encoder_weights", default='imagenet', type=str, help="model weights")
    parser.add_argument("--best_model_path", default='model/best_model.pth', type=str, help="path to save best model")
    parser.add_argument("--class_weight", default=50, type=int, help="balance factor")

    parser.add_argument("--loss_fn", default='bce', type=str, help="desired loss function")
    parser.add_argument("--opt", default='adam', type=str, help="desired optimizer")
    parser.add_argument("--scheduler", default='plat', type=str, help="desired lr scheduler")
    parser.add_argument("--patience", default=4, type=int, help="lr scheduler patience")
    parser.add_argument("--add_masks", default=50, type=int, help="non empty masks + N empty")

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
