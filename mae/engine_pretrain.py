# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import mae.util.logging as logging
import mae.util.lr_sched as lr_sched
import mae.util.misc as misc
import numpy as np
import torch
from iopath.common.file_io import g_pathmgr as pathmgr
from mae.util.misc import plot_input, get_mask_ratio
from timm.utils import accuracy


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    visualize=False,
    fp32=False,
):
    model.train(True)
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _, index) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        visualize_freq = 1
        mask_ratio = get_mask_ratio(args, epoch)
        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, _, _, vis = model(
                samples,
                index,
                mask_ratio=mask_ratio,
                visualize=visualize and data_iter_step % visualize_freq == 0,
            )

        if visualize and data_iter_step % visualize_freq == 0:
            if not pathmgr.exists(f"{args.output_dir}/vis95"):
                try:
                    pathmgr.mkdirs(f"{args.output_dir}/vis95")
                except Exception as e:
                    pass
            vis = vis.detach().cpu().permute(0, 1, 3, 2, 4, 5)
            for i in range(vis.shape[0]):
                # B 3 C T H W -> B 3 T C H W
                plot_input(
                    vis[i],
                    path=f"{args.output_dir}/vis95/{epoch}_{data_iter_step}_{misc.get_rank()}_{i}.jpg",
                    folder_path=f"{args.output_dir}/vis95/{epoch}_{data_iter_step}_{misc.get_rank()}_{i}",
                )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as e:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        # loss_scaler(
        #     loss,
        #     optimizer,
        #     parameters=model.parameters(),
        #     update_grad=(data_iter_step + 1) % accum_iter == 0,
        #     clip_grad=args.clip_grad,
        # )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=mask_ratio)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def knn_one_epoch(
    model: torch.nn.Module,
    data_loader_train: Iterable,
    data_loader_val: Iterable,
    device: torch.device,
    epoch: int,
    log_writer=None,
    args=None,
):
    model.eval()

    # Set env
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Knn trainset Epoch: [{}]".format(epoch)
    print_freq = 20
    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # Fill KNN bank from train set
    num_samples = len(data_loader_train.dataset)
    if args.distributed:
        embed_dim = model.module.embed_dim
    else:
        embed_dim = model.embed_dim
    stdv = 1.0 / math.sqrt(embed_dim / 3)
    knn_bank = torch.rand(num_samples, embed_dim).mul_(2 * stdv).add_(-stdv).to(device, non_blocking=True)

    train_labels = np.zeros((num_samples,), dtype=np.int32)
    for i in range(num_samples):
        train_labels[i] = data_loader_train.dataset._labels[i]
    train_labels = torch.LongTensor(train_labels).cuda()

    for data_iter_step, (samples, labels, index) in enumerate(
        metric_logger.log_every(data_loader_train, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        mask_ratio = get_mask_ratio(args, epoch)
        with torch.cuda.amp.autocast():
            knn_latent = model(
                samples,
                index,
                mask_ratio=mask_ratio,
                visualize=False,
                knn_only=True,
            )
            knn_bank.index_copy_(0, index, knn_latent)

        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())

    # Get KNN metric from val set
    knn_k = 200

    # Set env
    metric_logger = misc.MetricLogger(delimiter="  ")

    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    header = "Knn valset Epoch: [{}]".format(epoch)
    print_freq = 20
    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, labels, index) in enumerate(
        metric_logger.log_every(data_loader_val, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        mask_ratio = get_mask_ratio(args, epoch)
        with torch.cuda.amp.autocast():
            knn_latent = model(
                samples,
                index,
                mask_ratio=0.0,
                visualize=False,
                knn_only=True,
            )
            dist = torch.einsum("nc,mc->nm", knn_latent, knn_bank)
            yd, yi = dist.topk(knn_k, dim=1, largest=True, sorted=True)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = args.num_classes
            sigma = 1.0
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)

            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)

        acc1, acc5 = accuracy(preds, labels, topk=(1, 5))

        batch_size = preds.shape[0]
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5,
        )
    )
    if log_writer is not None:
        """We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        epoch_1000x = epoch * 1000 * args.repeat_aug
        log_writer.add_scalar("knn", acc1, epoch_1000x)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
