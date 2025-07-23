import math
import sys

import torch

import wandb

from .utils import MetricLogger, SmoothedValue, is_main_process, save_model


def train_action_vqvae(
    model: torch.nn.Module,
    model_dtype: str,
    train_dataloader,
    val_dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_scaler,
    clip_grad: float = 0,
    log_writer=None,
    lr_scheduler=None,
    lr_schedule_values=None,
    args=None,
    print_freq=20,
    total_steps=100000,
    model_without_ddp=None,
    initial_global_step: int = 0,
    add_validation=True,
):
    # The trainer for causal video vae
    train_dataloader_iterator = iter(train_dataloader)
    metric_logger = MetricLogger(delimiter="  ")
    if is_main_process():
        wandb.init(project="conv_vavae", name=args.wandb_name)

    if optimizer is not None:
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    if model_dtype == "bf16":
        _dtype = torch.bfloat16
    elif model_dtype == "fp32":
        _dtype = torch.float32
    else:
        _dtype = torch.float16

    val_steps = len(val_dataloader)
    train_steps = len(train_dataloader)
    for step in metric_logger.log_every(range(initial_global_step, total_steps), print_freq):
        if step >= total_steps:
            break

        model.train()
        if lr_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[step] * param_group.get("lr_scale", 1.0)

        batch = next(train_dataloader_iterator)  # B,5,7
        with torch.cuda.amp.autocast(enabled=True, dtype=_dtype):
            act = batch["actions"]
            commit_loss, rec_loss, total_loss = model(act)

        loss_log = {"commit_loss": commit_loss, "rec_loss": rec_loss}

        ###################################################################################################
        # The update of rec_loss
        loss_value = total_loss  # .item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        grad_norm = loss_scaler(
            total_loss,
            optimizer,
            clip_grad=clip_grad,
            parameters=model.module.vqvae.parameters(),
            create_graph=is_second_order,
        )
        if "scale" in loss_scaler.state_dict():
            loss_scale_value = loss_scaler.state_dict()["scale"]
        else:
            loss_scale_value = 1

        metric_logger.update(vae_loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        ###################################################################################################

        torch.cuda.synchronize()
        metric_logger.update(**loss_log)

        if total_loss is not None:
            min_lr = 10.0
            max_lr = 0.0
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

        if is_main_process():
            wandb.log(
                {
                    "train/commit_loss": commit_loss.item(),
                    "train/rec_loss": rec_loss.item(),
                    "train/total_loss": loss_value.item(),
                    "train/lr": max_lr,
                    "train/min_lr": min_lr,
                    "train/weight_decay": weight_decay_value,
                    "train/grad_norm": grad_norm,
                    "train/loss_scale": loss_scale_value,
                },
                step=initial_global_step,
            )

        if log_writer is not None:
            log_writer.update(**loss_log, head="train/loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(step)

        initial_global_step = initial_global_step + 1

        # gather the stats from all processes
        if step % print_freq == 0:
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)

        if step % args.save_ckpt_freq == 0:
            if is_main_process():
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    step=step,
                    save_ckpt_freq=args.save_ckpt_freq,
                )
        if add_validation:
            if (step + 1) % train_steps == 0:
                val_rec_loss = val_action_vqvae(model, val_dataloader, val_steps)
                if is_main_process():
                    wandb.log({"train/val_rec_loss": val_rec_loss}, step=args.global_step)


def val_action_vqvae(model, val_dataloader, val_steps):
    model.eval()
    total_val_loss = 0
    total_val_num_batch = 0
    for val_batch_idx, val_batch in enumerate(val_dataloader):
        val_act = val_batch["actions"]
        with torch.no_grad():
            val_commit_loss, val_rec_loss, val_total_loss = model(val_act)
        total_val_num_batch += val_act.shape[0]
        total_val_loss += val_rec_loss.mean().item() * val_act.shape[0]
        if (val_batch_idx + 1) == val_steps:
            break
    val_recon_loss = total_val_loss / total_val_num_batch
    return val_recon_loss
