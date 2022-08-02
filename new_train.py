# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

# import sacred
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler


from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.nn as nn
import wandb

wandb.init(project="DETR_obj_tracking", resume=False)
wandb.run.name = "Scratched_COCO_DETR_valid"
wandb.run.save()
import matplotlib.pyplot as plt


import trackformer.util.misc as utils
from trackformer.datasets import build_dataset
from trackformer.engine import evaluate, train_one_epoch
from trackformer.models import build_model
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.plot_utils import get_vis_win_names
from trackformer.vis import build_visualizers


def import_config():
    with open("/home/roberto/old_trackformer/cfgs/train_mot17.yaml", "r") as stream:
        try:
            train_mot17_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open("/home/roberto/old_trackformer/cfgs/train.yaml", "r") as stream:
        try:
            train_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open(
        "/home/roberto/old_trackformer/cfgs/train_deformable.yaml", "r"
    ) as stream:
        try:
            deformable_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open("/home/roberto/old_trackformer/cfgs/train_tracking.yaml", "r") as stream:
        try:
            tracking_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open("/home/roberto/old_trackformer/cfgs/train_mots20.yaml", "r") as stream:
        try:
            mots20_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open("/home/roberto/old_trackformer/cfgs/train.yaml", "r") as stream:
        try:
            full_res_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open("/home/roberto/old_trackformer/cfgs/train_full_res.yaml", "r") as stream:
        try:
            focal_loss_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return (
        train_yaml,
        deformable_yaml,
        tracking_yaml,
        mots20_yaml,
        full_res_yaml,
        focal_loss_yaml,
        train_mot17_yaml,
    )


def import_all_train_config():

    with open("/home/roberto/old_trackformer/cfgs/new_train.yaml", "r") as stream:
        try:
            train_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return train_yaml


loss_vec = []


def train(args: Namespace, obj_detect_checkpoint_file) -> None:
    print(args)

    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    if args.debug:
        # args.tracking_eval = False
        args.num_workers = 0

    if not args.deformable:
        assert args.num_feature_levels == 1
    if args.tracking:
        # assert args.batch_size == 1

        if args.tracking_eval:
            assert "mot" in args.dataset

    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(vars(args), open(output_dir / "config.yaml", "w"), allow_unicode=True)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model, criterion, postprocessors = build_model(args)

    ######### ######### added this part to load our model ######### #########
    args.use_excav_detr = False
    if args.use_excav_detr:
        if args.restart_training:
            print("\n reastarting training ")

            # our model
            obj_detect_checkpoint = torch.load(
                obj_detect_checkpoint_file, map_location=lambda storage, loc: storage
            )

            obj_detect_state_dict = obj_detect_checkpoint["model"]

            new_obj_detect_state_dict = obj_detect_state_dict.copy()
            for field in new_obj_detect_state_dict:
                if field[0:4] == "detr":
                    obj_detect_state_dict[field[5:]] = obj_detect_state_dict[field]
                    del obj_detect_state_dict[field]
                    new_field = field[5:]

            # load new layers
            track_att_checkpoint = torch.load(
                "models/mots20_train_masks/checkpoint.pth",
                map_location=lambda storage, loc: storage,
            )

            track_att_state_dict = track_att_checkpoint["model"]

            # add trackattention layers
            for keys in track_att_state_dict:
                if keys not in new_obj_detect_state_dict and keys[5:13] != "backbone":
                    # print("\n new key :", keys)
                    obj_detect_state_dict[keys] = track_att_state_dict[keys]

            obj_detect_state_dict = {
                k.replace("detr.", ""): v
                for k, v in obj_detect_state_dict.items()
                if "track_encoding" not in k
            }

            # for key in obj_detect_state_dict:
            #    print(key)

            detr_model = True
            if detr_model:

                weight_tensor = obj_detect_state_dict["class_embed.weight"][
                    :2, :
                ].clone()
                bias_tensor = obj_detect_state_dict["class_embed.bias"][:2].clone()

                del obj_detect_state_dict["class_embed.weight"]  # torch.Size([2, 256]).
                del obj_detect_state_dict["class_embed.bias"]  # torch.Size([2]).

                # obj_detect_state_dict["class_embed.weight"] = torch.zeros(2,256)
                # obj_detect_state_dict["class_embed.bias"] = torch.zeros(2)

                obj_detect_state_dict["class_embed.weight"] = weight_tensor
                obj_detect_state_dict["class_embed.bias"] = bias_tensor

            for k, v in obj_detect_state_dict.items():

                if "layers_track_attention" in k:
                    # print(obj_detect_state_dict[k].shape)
                    # for dim in obj_detect_state_dict[k].shape:
                    if (
                        len(obj_detect_state_dict[k].shape) <= 1
                    ):  # xavier initializer is not for bias
                        shape = obj_detect_state_dict[k].shape
                        obj_detect_state_dict[k] = torch.zeros(shape)
                    else:
                        nn.init.xavier_uniform_(obj_detect_state_dict[k])
                        # print(obj_detect_state_dict[k])

            model.load_state_dict(obj_detect_state_dict, strict=False)  # Change strict
            print(model)
        else:
            print("\n loading PRETRAINED model ")
            obj_detect_checkpoint = torch.load(
                obj_detect_checkpoint_file, map_location=lambda storage, loc: storage
            )

            obj_detect_state_dict = obj_detect_checkpoint["model"]
            model.load_state_dict(obj_detect_state_dict, strict=False)  # Change strict
    ######### ######### end of the code snippet to load our model ######### #########

    model.to(device)

    visualizers = build_visualizers(args)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("NUM TRAINABLE MODEL PARAMS:", n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(
                    n,
                    args.lr_backbone_names
                    + args.lr_linear_proj_names
                    + ["layers_track_attention"],
                )
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    if args.track_attention:
        param_dicts.append(
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, ["layers_track_attention"])
                    and p.requires_grad
                ],
                "lr": args.lr_track,
            }
        )

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop])

    dataset_train = build_dataset(split="train", args=args)
    dataset_val = build_dataset(split="val", args=args)

    if args.distributed:
        sampler_train = utils.DistributedWeightedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    best_val_stats = None
    if args.resume:

        checkpoint = torch.load(args.resume, map_location="cpu")

        model_state_dict = model_without_ddp.state_dict()
        checkpoint_state_dict = checkpoint["model"]

        checkpoint_state_dict = {
            k.replace("detr.", ""): v for k, v in checkpoint["model"].items()
        }

        if args.start_last_layers_from_scratch:
            for k, v in model_state_dict.items():
                if "class_embed" in k:
                    print("delete class_embed to start from scratch")
                    del checkpoint_state_dict[k]

        # delete class_embed.weight and bias since mismatching ADDED BY ME
        if not args.start_last_layers_from_scratch:
            for k, v in model_state_dict.items():
                if "class_embed" in k:
                    embed_weight_size = checkpoint_state_dict[k].shape
                    if embed_weight_size[0] == 1:
                        print("delete class_embed since weights are less then 2")
                        del checkpoint_state_dict[
                            "class_embed.weight"
                        ]  # torch.Size([2, 256]).
                        del checkpoint_state_dict[
                            "class_embed.bias"
                        ]  # torch.Size([2]).

        resume_state_dict = {}
        for k, v in model_state_dict.items():
            if k not in checkpoint_state_dict:
                resume_value = v
                print(f"Load {k} {tuple(v.shape)} from scratch.")
            elif v.shape != checkpoint_state_dict[k].shape:
                checkpoint_value = checkpoint_state_dict[k]
                num_dims = len(checkpoint_value.shape)

                if "norm" in k:
                    resume_value = checkpoint_value.repeat(2)
                elif "multihead_attn" in k or "self_attn" in k:
                    resume_value = checkpoint_value.repeat(num_dims * (2,))
                elif (
                    "reference_points" in k
                    and checkpoint_value.shape[0] * 2 == v.shape[0]
                ):
                    resume_value = v
                    resume_value[:2] = checkpoint_value.clone()
                elif "linear1" in k or "query_embed" in k:
                    resume_state_dict[k] = v
                    print(f"Load {k} {tuple(v.shape)} from scratch.")
                    continue
                elif "linear2" in k or "input_proj" in k:
                    resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1,))
                elif "class_embed" in k:
                    resume_value = checkpoint_value[[1]]
                    resume_value = v
                    print(f"Load {k} {tuple(v.shape)} from scratch.")
                else:
                    raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")

                print(
                    f"Load {k} {tuple(v.shape)} from resume model "
                    f"{tuple(checkpoint_value.shape)}."
                )
            elif args.resume_shift_neuron and "class_embed" in k:
                checkpoint_value = checkpoint_state_dict[k]
                # no-object class
                resume_value = checkpoint_value.clone()

                resume_value[:-1] = checkpoint_value[1:].clone()
                resume_value[-2] = checkpoint_value[0].clone()
                print(
                    f"Load {k} {tuple(v.shape)} from resume model and "
                    "shift class embed neurons to start with label=0 at neuron=0."
                )
            else:
                resume_value = checkpoint_state_dict[k]

            resume_state_dict[k] = resume_value

        if args.masks and args.load_mask_head_from_model is not None:
            checkpoint_mask_head = torch.load(
                args.load_mask_head_from_model, map_location="cpu"
            )

            for k, v in resume_state_dict.items():

                if (
                    "bbox_attention" in k or "mask_head" in k
                ) and v.shape == checkpoint_mask_head["model"][k].shape:
                    print(f"Load {k} {tuple(v.shape)} from mask head model.")
                    resume_state_dict[k] = checkpoint_mask_head["model"][k]

        model_without_ddp.load_state_dict(resume_state_dict, strict=True)

        # for k, v in resume_state_dict.items():
        #    if 'layers_track_attention' in k:
        #        print(resume_state_dict[k].shape)

        # RESUME OPTIM
        if not args.eval_only and args.resume_optim:
            if "optimizer" in checkpoint:
                if args.overwrite_lrs:
                    for c_p, p in zip(
                        checkpoint["optimizer"]["param_groups"], param_dicts
                    ):
                        c_p["lr"] = p["lr"]

                optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if "epoch" in checkpoint:
                args.start_epoch = checkpoint["epoch"] + 1

            best_val_stats = checkpoint["best_val_stats"]

        # RESUME VIS
        if not args.eval_only and args.resume_vis and "vis_win_names" in checkpoint:
            for k, v in visualizers.items():
                for k_inner in v.keys():
                    visualizers[k][k_inner].win = checkpoint["vis_win_names"][k][
                        k_inner
                    ]

    if args.eval_only:
        eval_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            device,
            output_dir,
            visualizers["val"],
            args,
        )

        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )

        return

    look_inside = False
    if look_inside:

        parameters = {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, ["embed"]) and p.requires_grad
            ]
        }

        print(
            f"\n those are the params : {parameters} and those their number \n \
        {sum(p.numel() for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, ['layers_track_attention']) and p.requires_grad)}"
        )

        return

    print("\n \n Start training \n \n ")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        # TRAIN
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # Modify the optimizer to use cosine annealing with warmup
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=200,
            cycle_mult=1.0,
            max_lr=0.0005,  # try with 10^-4, 5 * 10^-5 and 10^-5 as max values
            min_lr=args.lr,  # Min values an order of magnitude lower or quite smaller
            warmup_steps=100,
            gamma=1.0,
        )

        metrics, loss = train_one_epoch(
            model,
            criterion,
            postprocessors,
            data_loader_train,
            optimizer,
            device,
            epoch,
            visualizers["train"],
            scheduler,
            args,
        )

        args.eval_train = False  # change this overwrite
        if args.eval_train:
            random_transforms = data_loader_train.dataset._transforms
            data_loader_train.dataset._transforms = data_loader_val.dataset._transforms

            eval_stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_train,
                device,
                output_dir,
                visualizers["train"],
                args,
                epoch,
            )

            data_loader_train.dataset._transforms = random_transforms

        lr_scheduler.step()
        scheduler.step()

        checkpoint_paths = [output_dir / "checkpoint.pth"]

        validate = True
        if validate and epoch == 21:
            print("\n VALIDATION time of THE model ... \n")
            val_stats, _ = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                device,
                output_dir,
                visualizers["val"],
                args,
                epoch,
            )

        # VAL
        if False:  # epoch == 1 or not epoch % args.val_interval:
            print("\n VALIDATING THE model ... \n")
            val_stats, _ = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                device,
                output_dir,
                visualizers["val"],
                args,
                epoch,
            )

            checkpoint_paths = [output_dir / "checkpoint.pth"]

            # checkpoint for best validation stats
            stat_names = [
                "BBOX_AP_IoU_0_50-0_95",
                "BBOX_AP_IoU_0_50",
                "BBOX_AP_IoU_0_75",
            ]
            if args.masks:
                stat_names.extend(
                    ["MASK_AP_IoU_0_50-0_95", "MASK_AP_IoU_0_50", "MASK_AP_IoU_0_75"]
                )
            if args.tracking and args.tracking_eval:
                stat_names.extend(["MOTA", "IDF1"])

            if best_val_stats is None:
                best_val_stats = val_stats
            best_val_stats = [
                best_stat if best_stat > stat else stat
                for best_stat, stat in zip(best_val_stats, val_stats)
            ]
            for b_s, s, n in zip(best_val_stats, val_stats, stat_names):
                if b_s == s:
                    checkpoint_paths.append(output_dir / f"checkpoint_best_{n}.pth")

        # MODEL SAVING
        print("\n saving the model ... \n")
        if args.output_dir:
            if args.save_model_interval and not epoch % args.save_model_interval:
                checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch}.pth")

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "vis_win_names": get_vis_win_names(visualizers),
                        "best_val_stats": best_val_stats,
                    },
                    checkpoint_path,
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    train_yaml, deformable_yaml, tracking_yaml, mots20_yaml, full_res_yaml, focal_loss_yaml, train_mot17_yaml = (
        import_config()
    )

    # config = {**train_yaml, **deformable_yaml, **tracking_yaml, **mots20_yaml,
    # **full_res_yaml, **focal_loss_yaml} # dictionary concatenation

    """
    train_mots20 = False
    if train_mots20:
        config = {**train_yaml, **tracking_yaml, **mots20_yaml} # dictionary concatenation
        config["model_name"] = "ScratchedMots20Model"
        config["save_model_interval"] = 1

        wandb.config = config
        args = nested_dict_to_namespace(config)

        args.resume = "models/TrackformerDetrRetrained/checkpoint_epoch_6.pth"
        args.output_dir="/home/roberto/old_trackformer/models/" + config["model_name"]
        args.mot_path_train="data/MOTS20"
        args.mot_path_val="data/MOTS20"
        args.train_split="mots20_train_1_coco"
        args.val_split="mots20_val_1_coco"
        args.epochs= 21
        args.val_interval = 1
        args.start_last_layers_from_scratch = True
        args.backbone = "resnet101"

        train(args, None)

    else :
        config = {**train_yaml, **tracking_yaml, **mots20_yaml} # dictionary concatenation
        wandb.config = config

        # override configuration
        config["use_excav_detr"] = train_mots20
        config["model_name"] = "ScratchedExacvDetr"
        config["output_dir"] = "/home/roberto/old_trackformer/models/" + config["model_name"]
        config["eval_train"] = True
        config["save_model_interval"] = 5
        config["epochs"] = 21

        args = nested_dict_to_namespace(config)

        args.resume = "/home/roberto/old_trackformer/models/ExcavDETR_orig/detr_panoptic_model.pth"
        args.mot_path_train="data/MOTS20"
        args.mot_path_val="data/MOTS20"
        args.train_split="mots20_train_1_coco"
        args.val_split="mots20_val_1_coco"
        args.val_interval = 1
        args.num_workers = 0
        args.start_last_layers_from_scratch = False
        args.backbone = "resnet50"



        train(args, None)
    """
    train_fb_DETR = True
    if train_fb_DETR:
        config = {
            **train_yaml,
            **tracking_yaml,
            **mots20_yaml,
        }  # dictionary concatenation
        config["model_name"] = "COCO_DETR"
        config["save_model_interval"] = 1

        wandb.config = config
        args = nested_dict_to_namespace(config)

        args.resume = (
            "/home/roberto/old_trackformer/models/COCO_DETR/checkpoint_epoch_17.pth"
        )
        args.output_dir = "/home/roberto/old_trackformer/models/" + config["model_name"]
        args.mot_path_train = "data/MOTS20"
        args.mot_path_val = "data/MOTS20"
        args.train_split = "mots20_train_1_coco"
        args.val_split = "mots20_val_1_coco"
        args.epochs = 21
        args.val_interval = 1
        args.start_last_layers_from_scratch = False
        args.backbone = "resnet50"
        args.eval_only = True

        train(args, None)

    """
    python src/train.py with \
        mot17 \
        deformable \
        multi_frame \
        tracking \
        resume=models/mot17_crowdhuman_deformable_trackformer/checkpoint_epoch_40.pth \
        output_dir=models/custom_dataset_deformable \
        mot_path_train=data/custom_dataset \
        mot_path_val=data/custom_dataset \
        train_split=train \
        val_split=val \
        epochs=20 \
        """
