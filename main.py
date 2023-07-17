import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector',
                                     add_help=False)

    parser.add_argument('--num_global_frames', default=3, type=int)
    parser.add_argument('--self_attn', default=False, action='store_true')
    parser.add_argument('--shuffled_aug',
                        default='',
                        type=str,
                        help="Name of the shuffled video augmentation to use")
    parser.add_argument('--umap', default=False, action='store_true')
    parser.add_argument('--umap_loss', default=False, action='store_true')

    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names',
                        default=["backbone.0"],
                        type=str,
                        nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'],
                        type=str,
                        nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm',
                        default=0.1,
                        type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine',
                        default=False,
                        action='store_true')
    parser.add_argument('--bbox_gradient_propa', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument(
        '--frozen_weights',
        type=str,
        default=None,
        help=
        "Path to the pretrained model. If set, only the mask head will be trained"
    )

    # * Backbone
    parser.add_argument('--backbone',
                        default='resnet50',
                        type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument(
        '--dilation',
        action='store_true',
        help=
        "If true, we replace stride with dilation in the last convolutional block (DC5)"
    )
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features"
    )
    parser.add_argument('--position_embedding_scale',
                        default=2 * np.pi,
                        type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels',
                        default=4,
                        type=int,
                        help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers',
                        default=6,
                        type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers',
                        default=6,
                        type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument(
        '--dim_feedforward',
        default=1024,
        type=int,
        help=
        "Intermediate size of the feedforward layers in the transformer blocks"
    )
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout',
                        default=0.1,
                        type=float,
                        help="Dropout applied in the transformer")

    parser.add_argument(
        '--nheads',
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries',
                        default=300,
                        type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--dec_n_points_temporal_frame', default=4, type=int)
    parser.add_argument('--enc_n_points_temporal_frame', default=2, type=int)
    parser.add_argument('--enc_temporal_window', default=2, type=int)
    parser.add_argument('--enc_connect_all_frames', default=False, action='store_true')
    parser.add_argument('--instance_aware_attention', default=False, action='store_true')
    # * Segmentation
    parser.add_argument('--masks',
                        action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument(
        '--no_aux_loss',
        dest='aux_loss',
        action='store_false',
        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class',
                        default=2,
                        type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox',
                        default=5,
                        type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou',
                        default=2,
                        type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_mode', default='15frames', type=str)
    parser.add_argument('--data_coco_lite_path', default='./miccai_buv', type=str)

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--cache_mode',
                        default=False,
                        action='store_true',
                        help='whether to cache images on memory')
    parser.add_argument('--num_frames', default=3, type=int)
    parser.add_argument('--num_support_frames', default=2, type=int)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "config.txt").open("w") as f:
            f.write(str(args).replace(',', ',\n') + "\n")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val,
                                                          shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val,
                                                      shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                        args.batch_size,
                                                        drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [{
        "params": [
            p for n, p in model_without_ddp.named_parameters()
            if not match_name_keywords(n, args.lr_backbone_names)
            and not match_name_keywords(n, args.lr_linear_proj_names)
            and p.requires_grad
        ],
        "lr":
        args.lr,
    }, {
        "params": [
            p for n, p in model_without_ddp.named_parameters() if
            match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
        ],
        "lr":
        args.lr_backbone,
    }, {
        "params": [
            p for n, p in model_without_ddp.named_parameters()
            if match_name_keywords(n, args.lr_linear_proj_names)
            and p.requires_grad
        ],
        "lr":
        args.lr * args.lr_linear_proj_mult,
    }]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts,
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

        
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device,
                                              args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval,
                                 output_dir / "eval.pth")
        return

    print("Start training")
    writer = SummaryWriter(args.output_dir + '/summary/')

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train,
                                      optimizer, device, epoch,
                                      args.clip_max_norm, writer)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir /
                                        f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            # tensorboard
            writer.add_scalar('loss', train_stats['loss'], global_step=epoch)
            writer.add_scalar('loss_bbox',
                              train_stats['loss_bbox'],
                              global_step=epoch)
            writer.add_scalar('loss_ce',
                              train_stats['loss_ce'],
                              global_step=epoch)
            writer.add_scalar('loss_giou',
                              train_stats['loss_giou'],
                              global_step=epoch)
            writer.add_scalar('lr', train_stats['lr'], global_step=epoch)
            writer.add_scalar('class_error',
                              train_stats['class_error'],
                              global_step=epoch)

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device,
                                              args.output_dir)

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                log_stats = str(json.dumps(log_stats)).replace(
                    '"test_coco_eval_bbox"', '\n"test_coco_eval_bbox"')
                f.write(log_stats + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script',
        parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
