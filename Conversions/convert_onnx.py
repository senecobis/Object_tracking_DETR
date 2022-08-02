import torch.onnx
import onnxruntime as rt
import onnx

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

import argparse, os
import yaml


""" 
    run this script to convert your pytorch model specified in directory     
    path = detr_config.frozen_weights into open neural network exchange
    format
"""


""" Source the parent directory.
 for istance you can run 

 PYTHONPATH=$PYTHONPATH:/path/to/your/cloned/repo   
 export PYTHONPATH

 PYTHONPATH=$PYTHONPATH:/home/rpellerito/project_folder/catkin_ws/src/m545_panoptic_segmentation   
 export PYTHONPATH
 """


def get_args_parser():
    parser = argparse.ArgumentParser("Convert settings", add_help=False)
    parser.add_argument(
        "--model_input_dimension",
        default=(1, 3, 1376, 1152),
        type=tuple,
        help="specify it as a tuple : (1,3,-,-,)",
    )
    parser.add_argument("--output_folder", default="out", type=str)
    parser.add_argument("--model_name", default="new_model.onnx", type=str)

    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="opset value of onnx converter (specify the latest)",
    )

    # additional settings
    parser.add_argument("--do_constant_folding", default=True, type=bool)
    parser.add_argument("--export_params", default=True, type=bool)

    parser.add_argument("--verbose", default=True, type=bool)

    return parser


# Function to Convert our pytorch model to ONNX
def Convert_ONNX(args):
    # Onnx needs an input to see how the input tensor propagates through the model,
    model_input_size = (
        torch.randn(args.model_input_dimension).to("cuda").to(torch.float32)
    )

    # load the pytorch model

    path = "/home/roberto/old_trackformer/models/mots20_train_masks/checkpoint.pth"

    obj_detect_config_path = os.path.join(os.path.dirname(path), "config.yaml")

    obj_detect_args = nested_dict_to_namespace(
        yaml.unsafe_load(open(obj_detect_config_path))
    )

    model, _, _ = build_model(obj_detect_args)
    checkpoint = torch.load(path, map_location="cuda")
    obj_detect_state_dict = checkpoint["model"]

    obj_detect_state_dict = {
        k.replace("detr.", ""): v
        for k, v in obj_detect_state_dict.items()
        if "track_encoding" not in k
    }  # this should delete detr. but it doesn't work

    model.load_state_dict(obj_detect_state_dict)
    model.cuda()

    # model, _ = PanopticNets.DETR(panoptic_models.detr.config)._load_model_and_postprocessor()
    # path = detr_config.frozen_weights
    # checkpoint = torch.load(path, map_location='cuda')
    # model.load_state_dict(checkpoint['model'])

    # set the model to inference mode
    model.eval()
    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    model_folder_path = os.path.abspath(args.output_folder)

    model_path = model_folder_path + "/" + args.model_name

    torch.onnx.export(
        model,
        model_input_size,
        model_path,  # where to save the model
        output_names=["output"],
        opset_version=args.opset_version,  # the ONNX version to export the model to
        do_constant_folding=args.do_constant_folding,  # whether to execute constant folding for optimization
        export_params=args.export_params,
    )  # store the trained parameter weights inside the model file

    print("\n--> Model has been converted to ONNX")

    # load onnx model
    model = onnx.load(model_path)

    # check if the IR is well formed
    onnx.checker.check_model(model)
    print("\n --> Model is well formed ")

    # print a readable representation of the model as a graph
    if args.verbose:
        print(
            "graph representation of the model : ",
            onnx.helper.printable_graph(model.graph),
        )


if __name__ == "__main__":
    # execute conversion
    parser = argparse.ArgumentParser(
        "ONNX converting script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Convert_ONNX(args)
