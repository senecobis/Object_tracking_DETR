# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from email.policy import strict
import os
from pickle import NONE
import sys
import time
from os import path as osp
from unittest import result
from PIL import Image

import motmetrics as mm
import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader
import trackformer.util.misc as utils


from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.track_utils import (
    evaluate_mot_accums,
    get_mot_accum,
    interpolate_tracks,
    plot_sequence,
)

import timeit, time


""" Source the parent directory.
 for istance you can run 

 PYTHONPATH=$PYTHONPATH:/path_to/trackformer
 export PYTHONPATH

 Then.
 Run python src/new_track.py 
 """

mm.lap.default_solver = "lap"


def main(
    seed,
    dataset_name,
    obj_detect_checkpoint_file,
    tracker_cfg,
    write_images,
    output_dir,
    interpolate,
    verbose,
    load_results_dir,
    data_root_dir,
    generate_attention_maps,
    frame_range,
    _config,
    obj_detector_model,
):

    if write_images:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during
    # training. in that case we omit verbose outputs.

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        yaml.dump(
            _config,
            open(osp.join(output_dir, "track.yaml"), "w"),
            default_flow_style=False,
        )

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    if obj_detector_model is None:
        obj_detect_config_path = os.path.join(
            os.path.dirname(obj_detect_checkpoint_file), "config.yaml"
        )

        obj_detect_args = nested_dict_to_namespace(
            yaml.unsafe_load(open(obj_detect_config_path))
        )
        img_transform = obj_detect_args.img_transform

        obj_detector, _, obj_detector_post = build_model(obj_detect_args)

        # our model
        obj_detect_checkpoint = torch.load(
            obj_detect_checkpoint_file, map_location=lambda storage, loc: storage
        )

        obj_detect_state_dict = obj_detect_checkpoint["model"]

        obj_detector.load_state_dict(
            obj_detect_state_dict, strict=False
        )  # Change strict

        # Object detector is the model
        # print("\n the obj_detector", obj_detector)

        if "epoch" in obj_detect_checkpoint:
            print(f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]")

        obj_detector.cuda()
    else:
        obj_detector = obj_detector_model["model"]
        obj_detector_post = obj_detector_model["post"]
        img_transform = obj_detector_model["img_transform"]

    if hasattr(obj_detector, "tracking"):
        obj_detector.tracking()

    track_logger = None
    tracker = Tracker(
        obj_detector,
        obj_detector_post,
        tracker_cfg,
        generate_attention_maps,
        track_logger,
    )

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = TrackDatasetFactory(
        dataset_name, root_dir=data_root_dir, img_transform=img_transform
    )
    print("\n dataset", dataset.__len__())
    for seq in dataset:
        tracker.reset()

        print(f"TRACK SEQ: {seq}")

        # frame 380 person is visible
        start_frame = int(frame_range["start"] * len(seq))
        print("\n", start_frame)

        end_frame = int(frame_range["end"] * len(seq))
        print("\n", end_frame)

        seq_loader = DataLoader(
            torch.utils.data.Subset(seq, range(start_frame, end_frame))
        )

        num_frames += len(seq_loader)

        results = seq.load_results(load_results_dir)

        if not results:
            start = time.time()

            time_vec = []
            for frame_id, frame_data in enumerate(
                tqdm.tqdm(seq_loader, file=sys.stdout)
            ):
                start = time.time()
                with torch.no_grad():
                    tracker.step(frame_data)
                torch.cuda.synchronize()
                end = time.time() - start
                time_vec.append(end)

            print("time average", np.mean(time_vec))
            print("time variance", np.var(time_vec))

            results = tracker.get_results()
            time_total += time.time() - start

            print(f"NUM TRACKS: {len(results)} ReIDs: {tracker.num_reids}")
            print(f"RUNTIME: {time.time() - start :.2f} s")

            if interpolate:
                results = interpolate_tracks(results)

            if output_dir is not None:
                print(f"WRITE RESULTS")
                seq.write_results(results, output_dir)

        if seq.no_gt:
            print(seq.no_gt)
            print("NO GT AVAILBLE")
        else:
            print(np.size(results))
            mot_accum = get_mot_accum(results, seq_loader)
            mot_accums.append(mot_accum)

            if verbose:
                mot_events = mot_accum.mot_events
                reid_events = mot_events[mot_events["Type"] == "SWITCH"]
                match_events = mot_events[mot_events["Type"] == "MATCH"]

                switch_gaps = []
                for index, event in reid_events.iterrows():
                    frame_id, _ = index
                    match_events_oid = match_events[match_events["OId"] == event["OId"]]
                    match_events_oid_earlier = match_events_oid[
                        match_events_oid.index.get_level_values("FrameId") < frame_id
                    ]

                    if not match_events_oid_earlier.empty:
                        match_events_oid_earlier_frame_ids = match_events_oid_earlier.index.get_level_values(
                            "FrameId"
                        )
                        last_occurrence = match_events_oid_earlier_frame_ids.max()
                        switch_gap = frame_id - last_occurrence
                        switch_gaps.append(switch_gap)

                switch_gaps_hist = None
                if switch_gaps:
                    switch_gaps_hist, _ = np.histogram(
                        switch_gaps, bins=list(range(0, max(switch_gaps) + 10, 10))
                    )
                    switch_gaps_hist = switch_gaps_hist.tolist()

        if output_dir is not None and write_images:
            plot_sequence(
                results,
                seq_loader,
                osp.join(output_dir, dataset_name, str(seq)),
                write_images,
                generate_attention_maps,
            )

    if obj_detector_model is None:

        summary, str_summary = evaluate_mot_accums(
            mot_accums, [str(s) for s in dataset if not s.no_gt]
        )
        print("\n summary", summary)

        return summary

    return mot_accums


if __name__ == "__main__":
    working_dir = os.getcwd()
    with open(working_dir + "/cfgs/track.yaml", "r") as stream:
        try:
            track_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(
        dataset_name="EXCAV",
        data_root_dir=working_dir + "/data/Multi_pp_excav_flipped",
        output_dir="data/outdir/Multi_pp_excav",
        write_images="pretty",
        seed=666,
        interpolate=False,
        verbose=True,
        load_results_dir=None,
        generate_attention_maps=False,
        tracker_cfg=track_yaml["tracker_cfg"],
        obj_detect_checkpoint_file=working_dir + "models/ExacavDETR_101_20epochs/checkpoint_excavDETR_epoch_21.pth",
        frame_range={"start": 0.0, "end": 1.0},
        _config="cfgs/track.yaml",
        obj_detector_model=None,
    )
