import os
import sys

sys.path.append(os.path.abspath("."))
import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from torch.utils.data import DataLoader

import prismatic.util.vae_utils as utils
from prismatic.action_vqvae import ActionVQVAELossWrapper
from prismatic.trainer_misc import (
    NativeScalerWithGradNormCount,
    auto_load_model,
    cosine_scheduler,
    create_optimizer,
    init_distributed_mode,
    save_model,
    train_action_vqvae,
)
from prismatic.vla.datasets import RLDSActionBatchTransform, VqVAERLDSDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser("Pytorch Multi-process Training script for Video VAE", add_help=False)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--save_ckpt_freq", default=20, type=int)

    # Model parameters
    parser.add_argument("--ema_update", action="store_true")
    parser.add_argument("--ema_decay", default=0.99, type=float, metavar="MODEL", help="ema decay for quantizer")

    parser.add_argument("--vqvae_config_path", default="", type=str, help="The vae weight path")
    parser.add_argument("--model_dtype", default="bf16", help="The Model Dtype: bf16 or df16")

    # Optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw"')
    parser.add_argument(
        "--opt_eps", default=1e-8, type=float, metavar="EPSILON", help="Optimizer Epsilon (default: 1e-8)"
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay (default: 1e-4)")
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)""",
    )

    parser.add_argument("--lr", type=float, default=5e-5, metavar="LR", help="learning rate (default: 5e-5)")
    parser.add_argument(
        "--warmup_lr", type=float, default=1e-6, metavar="LR", help="warmup learning rate (default: 1e-6)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0 (1e-5)"
    )

    parser.add_argument(
        "--total_steps", type=int, default=100000, metavar="N", help="total steps to train, if scheduler supports"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, metavar="N", help="steps to warmup LR, if scheduler supports"
    )

    # Dataset parameters
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument(
        "--image_mix_ratio", default=0.1, type=float, help="The image data proportion in the training batch"
    )

    # Distributed Training parameters
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(no_auto_resume=True)

    parser.add_argument("--dist_eval", action="store_true", default=True, help="Enabling distributed evaluation")
    parser.add_argument("--disable_eval", action="store_true", default=False)

    parser.add_argument("--eval", action="store_true", default=False, help="Perform evaluation only")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--global_step", default=0, type=int, metavar="N", help="The global optimization step")
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--data_root_dir", default="", help="path where datasets are placed")
    parser.add_argument("--train_dataset_name", default="libero_90_no_noops", help="the name of dataset")
    parser.add_argument("--val_dataset_name", default="libero_90_no_noops", help="the name of valiation dataset")

    parser.add_argument("--wandb_name", default="test", help="the name of wandb")

    parser.add_argument("--use_action_type_pe", action="store_true", default=False, help="use action type pe")
    parser.add_argument("--use_time_pe", action="store_true", default=False, help="use time pe")

    parser.add_argument("--add_validation", action="store_true", default=False, help="add validation")
    parser.add_argument("--checkpoint_path", default=None, help="the path of checkpoint")
    parser.add_argument("--resume_vqvae", action="store_true", default=False, help="Resume VQVAE from a checkpoint. The starting point is determined by the checkpoint filename format: checkpoint-{steps}.pth")
    return parser.parse_args()


def build_model(args):
    vqvae_config_path = args.vqvae_config_path

    model = ActionVQVAELossWrapper(
        vqvae_config_path,
        use_action_type_pe=args.use_action_type_pe,
        use_time_pe=args.use_time_pe,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume_vqvae,
    )
    return model


# fix the seed for reproducibility
def seed_everything(seed):
    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main(args):
    init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    seed_everything(args.seed)
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(args)

    log_writer = None

    batch_transform = RLDSActionBatchTransform()

    training_dataset = VqVAERLDSDataset(
        args.data_root_dir,
        args.train_dataset_name,
        batch_transform,
        window_size=5,
        shuffle_buffer_size=200000,
        only_action=True,
        sample_ratio=1,
    )

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        sampler=None,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    vla_val_dataset = VqVAERLDSDataset(
        args.data_root_dir,
        args.val_dataset_name,
        batch_transform,
        window_size=5,
        shuffle_buffer_size=200000,
        only_action=True,
    )

    val_dataloader = DataLoader(
        vla_val_dataset,
        batch_size=args.batch_size,
        sampler=None,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    torch.distributed.barrier()

    model.to(device)
    model_without_ddp = model

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(name)
    print(f"total number of learnable params: {n_learnable_parameters / 1e6} M")
    print(f"total number of fixed params in : {n_fix_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weigth Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)

    optimizer = create_optimizer(args, model_without_ddp.vqvae)

    loss_scaler = NativeScalerWithGradNormCount(enabled=True if args.model_dtype == "fp16" else False)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Use step level LR & WD scheduler!")

    lr_schedule_values = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.total_steps,
        warmup_steps=args.warmup_steps,
    )

    auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"The global iterations is {args.global_step}")
    start_time = time.time()
    torch.distributed.barrier()

    if args.resume_vqvae:
        filename = os.path.basename(args.checkpoint_path)
        initial_global_step = int(filename.split("-")[-1].replace(".pth", "")) + 1
    else:
        initial_global_step = 0

    train_action_vqvae(
        model,
        args.model_dtype,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        loss_scaler,
        args.clip_grad,
        log_writer=log_writer,
        lr_schedule_values=lr_schedule_values,
        args=args,
        print_freq=args.print_freq,
        total_steps=args.total_steps,
        model_without_ddp=model_without_ddp,
        add_validation=args.add_validation,
        initial_global_step=initial_global_step,
    )

    if args.output_dir:
        save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            step=args.total_steps,
            save_ckpt_freq=args.save_ckpt_freq,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
