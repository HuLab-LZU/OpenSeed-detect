import glob
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Any

import pandas as pd
import regex
import rootutils
import yaml
from tqdm import tqdm

rootutils.setup_root(".", indicator=".project-root-yolo26", pythonpath=True, cwd=True)

# YOLO class will process YOLOE and YOLOWorld models internally so here we do not need to import them
from ultralytics.models import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, nms, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import Metric

LOG_DIR = Path("../../runs/detect")

MODEL_VARIANTS = [
    # "yolov5n",
    # "yolov5s",
    # "yolov5m",
    # "yolov5l",
    # "yolov5x",
    # #
    # "yolov8n-worldv2",
    # "yolov8s-worldv2",
    # "yolov8m-worldv2",
    # "yolov8l-worldv2",
    # "yolov8x-worldv2",
    # #
    # "yolov8n",
    # "yolov8s",
    # "yolov8m",
    # "yolov8l",
    # "yolov8x",
    #
    # "yolo11n",
    # "yolo11s",
    # "yolo11m",
    # "yolo11l",
    # "yolo11x",
    #
    # "yolov13n",
    # "yolov13s",
    # "yolov13l",
    # "yolov13x",
    #
    # "yoloe-11n",
    # "yoloe-11s",
    # "yoloe-11m",
    # "yoloe-11l",
    # "yoloe-11x",
    #
    # "yolo-master-n",
    # "yolo-master-s",
    # "yolo-master-m",
    # "yolo-master-l",
    # "yolo-master-x",
    #
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
    #
    # "rt-detr-s",
    # "rt-detr-m",
    # "rt-detr-l",
    # "rt-detr-x",
]

# runs/detect/yolo11l-detect-e100-b16-sz640-cp0.0-optSGD-s21-
EXPERIMENT_DIR_REGEX = regex.compile(
    r"^(?P<model_variant>{})"
    r"-(?P<experiment_name>.+)"
    r"-s(?P<seed>\d+)-$".format("|".join(MODEL_VARIANTS))
)
DATASET_YML = "cfg/datasets/OpenSeed-LZU-detect.yaml"

arg_keys = [
    "batch",
    "imgsz",
    "optimizer",
    "iou",
    "max_det",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "nbs",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
    "erasing",
]
header = [
    "model_variant",
    "experiment_name",
    "seed",
    "precision(B)",
    "recall(B)",
    "mAP50(B)",
    "mAP75(B)",
    "mAP50-95(B)",
    "mAP50-95(Small)",
    "mAP50-95(Medium)",
    "mAP50-95(Large)",
    "fitness",
    *arg_keys,
]


class OMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.size_specific_metrics = [0.0, 0.0, 0.0]  # [map_small, map_medium, map_large]

    @property
    def map_small(self) -> float:
        """
        Return the mean Average Precision (mAP) for small objects at IoU threshold of 0.5-0.95.

        Returns:
            (float): The mAP for small objects, or 0.0 if not available.
        """
        return self.size_specific_metrics[0]

    @property
    def map_medium(self) -> float:
        """
        Return the mean Average Precision (mAP) for medium objects at IoU threshold of 0.5-0.95.

        Returns:
            (float): The mAP for medium objects, or 0.0 if not available.
        """
        return self.size_specific_metrics[1]

    @property
    def map_large(self) -> float:
        """
        Return the mean Average Precision (mAP) for large objects at IoU threshold of 0.5-0.95.

        Returns:
            (float): The mAP for large objects, or 0.0 if not available.
        """
        return self.size_specific_metrics[2]

    def update_size_metrics(self, map_small: float, map_medium: float, map_large: float) -> None:
        """
        Update size-specific mAP metrics.

        Args:
            map_small (float): mAP for small objects.
            map_medium (float): mAP for medium objects.
            map_large (float): mAP for large objects.
        """
        print("UPDATE SIZE METRICS")
        self.size_specific_metrics = [map_small, map_medium, map_large]


class OSeedDetectionValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.metrics.box = OMetric()

    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self.is_coco = True

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        # predictions.json is coco-format and the categories are plused by 1: 0 is background
        # so instances_coco_test_start_1.json should start with 1
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = self.data["path"] / "annotations" / "instances_coco_test_start_1.json"
        ranges = {"small": [0, 25**2], "medium": [25**2, 125**2], "large": [125**2, 100000**2]}
        return self.coco_evaluate(stats, pred_json, anno_json, ranges=ranges)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
        ranges: dict | None = {"small": [0, 32**2], "medium": [32**2, 96**2], "large": [96**2, 100000**2]},
    ) -> dict[str, Any]:
        """
        Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics
        for object detection. Updates the provided stats dictionary with computed metrics
        including mAP50, mAP50-95, and LVIS-specific metrics if applicable.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path]): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path]): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | list[str]]): IoU type(s) for evaluation. Can be single string or list of strings.
                Common values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | list[str]]): Suffix to append to metric names in stats dictionary. Should correspond
                to iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno,
                        pred,
                        iouType=iou_type,
                        ranges=ranges,
                        lvis_style=self.is_lvis,
                        print_function=LOGGER.info,
                    )
                    val.params.imgIds = [
                        int(Path(x).stem) for x in self.dataloader.dataset.im_files
                    ]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    # Capture size-specific metrics when available (COCO only)
                    if self.is_coco and hasattr(val, "stats") and len(val.stats) >= 6:
                        # COCO stats array: [AP@50:95, AP@50, AP@75, AP@50:95 (small), AP@50:95 (medium), AP@50:95 (large), AR50-95, ..]
                        self.metrics.box.update_size_metrics(
                            map_small=val.stats[3],  # AP for small objects
                            map_medium=val.stats[4],  # AP for medium objects
                            map_large=val.stats[5],  # AP for large objects
                        )

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats


def eval_one_model(
    proj_dir: Path,
    overwrite: bool,
    dataset_yml: str,
    batch_size: int,
    device: list[int],
    try_end2end: bool = False,
):
    match = EXPERIMENT_DIR_REGEX.match(proj_dir.name)
    if match is None:
        return
    print(f"processing {proj_dir}")
    match_groups = match.groupdict()

    with open(proj_dir / "args.yaml") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    out_dir = proj_dir / "test"
    out_results = out_dir / "results.json"
    if out_results.exists() and not overwrite:
        with open(out_results, "r") as f:
            results_dict = json.load(f)
    else:
        model_path = proj_dir / "weights" / "best.pt"
        model = YOLO(model_path).eval()
        if try_end2end:
            results = model.val(
                validator=OSeedDetectionValidator,
                name=proj_dir.name + "/test",
                data=dataset_yml,
                split="test",
                imgsz=args["imgsz"],
                batch=batch_size,
                device=device,
                verbose=False,
                exist_ok=True,
                save_json=True,
                end2end="yolo26" in proj_dir.name or "yolo10" in proj_dir.name,
            )
        else:
            results = model.val(
                validator=OSeedDetectionValidator,
                name=proj_dir.name + "/test",
                data=dataset_yml,
                split="test",
                imgsz=args["imgsz"],
                batch=batch_size,
                device=device,
                verbose=False,
                exist_ok=True,
                save_json=True,
            )
        results_dict = results.results_dict
        results_dict["metrics/mAP75(B)"] = float(results.box.map75)
        results_dict["metrics/mAP50-95(Small)"] = float(results.box.map_small)
        results_dict["metrics/mAP50-95(Medium)"] = float(results.box.map_medium)
        results_dict["metrics/mAP50-95(Large)"] = float(results.box.map_large)
        with open(out_results, "w") as f:
            json.dump(results_dict, f, indent=4)
    line = [
        match_groups["model_variant"],
        match_groups["experiment_name"],
        int(match_groups["seed"]),
        results_dict["metrics/precision(B)"],
        results_dict["metrics/recall(B)"],
        results_dict["metrics/mAP50(B)"],
        results_dict["metrics/mAP75(B)"],
        results_dict["metrics/mAP50-95(B)"],
        results_dict["metrics/mAP50-95(Small)"],
        results_dict["metrics/mAP50-95(Medium)"],
        results_dict["metrics/mAP50-95(Large)"],
        results_dict["fitness"],
        *[args[k] for k in arg_keys],
    ]
    return line


def eval_all_models(args: Namespace):
    output_dir: Path = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    all_results = []
    experiment_dirs = glob.glob(str(log_dir / "*"))
    for d in tqdm(experiment_dirs):
        line = eval_one_model(
            Path(d),
            args.overwrite,
            args.dataset,
            args.batch_size,
            args.device,
            try_end2end=args.try_end2end,
        )
        if line is None:
            continue
        all_results.append(line)

    df = pd.DataFrame(all_results, columns=header)
    df.sort_values(by=["model_variant", "seed"], ascending=True, inplace=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "detect_results.csv", index=False)


def main(args: Namespace):
    eval_all_models(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset", type=str, default="cfg/datasets/OpenSeed-LZU-detect.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--log-dir", type=str, default="runs/detect")
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    parser.add_argument("--try-end2end", action="store_true")

    args = parser.parse_args()
    main(args)
