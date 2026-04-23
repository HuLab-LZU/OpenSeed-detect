"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import warnings
from pathlib import Path

import onnx
import onnxruntime as ort
import torchvision
from onnx import helper, shape_inference
from onnxruntime.tools.onnx_model_utils import optimize_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.core import YAMLConfig, register

warnings.filterwarnings("ignore")


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DeployPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        labels, scores = None, None
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        return labels, boxes, scores


def force_inference_onnx_shape(model, output_file):
    model.graph.ClearField("value_info")
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, output_file)

    return inferred_model


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    class ModelNew(nn.Module):
        def __init__(self, num_classes=656, imgsz=640, use_focal_loss=True) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.imgsz = imgsz
            self.model = cfg.model.deploy()
            self.postprocessor = DeployPostProcessor(
                num_classes=num_classes,
                use_focal_loss=use_focal_loss,
            ).eval()

        def forward(self, images):
            outputs = self.model(images)

            # logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
            # outputs = torch.cat([boxes, logits], dim=-1)
            # print(logits.shape, boxes.shape, outputs.shape)
            # return outputs

            labels, boxes, scores = self.postprocessor(outputs)
            print(labels.shape, boxes.shape, scores.shape)
            # keep same as yolo26, x1,y1,x2,y2,confidence,class
            # scale boxes to image size, this is the behavior of yolo26
            outputs = torch.cat([boxes * self.imgsz, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
            print(outputs.shape)
            # raise NotImplementedError("")
            return outputs

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    size = torch.tensor([[640, 640]])
    dummy_input = torch.rand(4, 3, 640, 640) if args.dynamic else torch.rand(1, 3, 640, 640)
    model = None
    export_args = None
    input_names=["images"]
    output_names = []
    dynamic_shapes = {
            "images": {
                0: "N",
            },
        }
    if args.yolo26_like_output:
        model = ModelNew().eval()
        export_args = (dummy_input,)
        output_names.append("output0")
        _ = model(dummy_input)
    else:
        model = Model().eval()
        export_args = (dummy_input, size)
        dynamic_shapes["orig_target_sizes"] = {0: "N"}
        input_names.append("orig_target_sizes")
        output_names.extend(["labels", "boxes", "scores"])
        _ = model(dummy_input, size)

    pth_model, pth_seed, _ = args.resume.split("/")[-3:]
    out_model_name = pth_model.replace("rtv4_hgnetv2_", "rt-detrv4-").replace("_coco", "") + f"_{pth_seed}"
    output_file = os.path.join(args.output_dir, f"{out_model_name}.onnx")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.onnx.export(
        model,
        export_args,
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        external_data=False,
        dynamo=args.dynamic,
        dynamic_shapes=dynamic_shapes if args.dynamic else None,
        opset_version=16,
        verbose=False,
        do_constant_folding=False,
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        try:
            onnx.checker.check_model(onnx_model, full_check=True)
        except Exception as e:
            print(f"Check export onnx model failed, trying to fix model shape inference, error: {e}")
            # force_inference_onnx_shape(onnx_model, output_file)
            optimize_model(
                Path(output_file),
                Path(output_file),
                level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            )

        print("output shape:", onnx_model.graph.output[0].type.tensor_type.shape.dim)
        print(f"Check export onnx model done, exported to {output_file}")

    if args.simplify:
        import onnx
        import onnxsim

        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = None
        if args.yolo26_like_output:
            input_shapes = (
                {
                    "images": dummy_input.shape,
                }
                if args.dynamic
                else None
            )
        else:
            input_shapes = (
                {"images": dummy_input.shape, "orig_target_sizes": size.shape} if args.dynamic else None
            )
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplify onnx model, status: {check}, save to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/dfine/dfine_hgnetv2_l_coco.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--yolo26-like-output",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_models",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    main(args)
