import warnings
from pathlib import Path
from typing import Literal, TypeAlias

import rootutils
import torch  # noqa: F401
from tap import Tap

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from ultralytics import RTDETR, YOLO, YOLOE, YOLOWorld
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOEPETrainer
from ultralytics.utils import RUNS_DIR

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

Optimizers: TypeAlias = Literal[
    "SGD",
    "Adam",
    "Adamax",
    "AdamW",
    "NAdam",
    "RAdam",
    "RMSProp",
    "auto",
]

OpenSeedDatasets: TypeAlias = Literal[
    "OpenSeed-LZU-detect.yaml",
    "OpenSeed-LZU-seg.yaml",
    "OpenSeed-LZU-detect-world.yaml",
    "OpenSeed-LZU-seg-world.yaml",
    "Objects365.yaml",
]

MODEL_WEIGHTS = {
    # YOLO v5
    "yolov5n.yaml": "weights/yolov5nu.pt",
    "yolov5s.yaml": "weights/yolov5su.pt",
    "yolov5m.yaml": "weights/yolov5mu.pt",
    "yolov5l.yaml": "weights/yolov5lu.pt",
    "yolov5x.yaml": "weights/yolov5xu.pt",
    # YOLO v8
    "yolov8n.yaml": "weights/yolov8n.pt",
    "yolov8s.yaml": "weights/yolov8s.pt",
    "yolov8m.yaml": "weights/yolov8m.pt",
    "yolov8l.yaml": "weights/yolov8l.pt",
    "yolov8x.yaml": "weights/yolov8x.pt",
    # YOLO 11
    "yolo11n.yaml": "weights/yolo11n.pt",
    "yolo11s.yaml": "weights/yolo11s.pt",
    "yolo11m.yaml": "weights/yolo11m.pt",
    "yolo11l.yaml": "weights/yolo11l.pt",
    "yolo11x.yaml": "weights/yolo11x.pt",
    # YOLO v13
    "yolov13n.yaml": "weights/yolov13n.pt",
    "yolov13s.yaml": "weights/yolov13s.pt",
    "yolov13l.yaml": "weights/yolov13l.pt",
    "yolov13x.yaml": "weights/yolov13x.pt",
    # YOLO E, n and x are not available since no pretrained weights
    # "yoloe-11n.yaml": "weights/yoloe-11n.pt",
    "yoloe-11s.yaml": "weights/yoloe-11s-seg.pt",
    "yoloe-11m.yaml": "weights/yoloe-11m-seg.pt",
    "yoloe-11l.yaml": "weights/yoloe-11l-seg.pt",
    # "yoloe-11x.yaml": "weights/yoloe-11x-seg.pt",
    # YOLO World
    "yolov8s-worldv2.yaml": "weights/yolov8s-worldv2.pt",
    "yolov8m-worldv2.yaml": "weights/yolov8m-worldv2.pt",
    "yolov8l-worldv2.yaml": "weights/yolov8l-worldv2.pt",
    "yolov8x-worldv2.yaml": "weights/yolov8x-worldv2.pt",
    # RTDETR
    "rtdetr-l.yaml": "weights/rtdetr-l.pt",
    "rtdetr-x.yaml": "weights/rtdetr-x.pt",
}


class OpenSeedArgs(Tap):
    model: str = "yolo11n.yaml"
    task: Literal["detect", "segment"] = "detect"
    epochs: int = 200
    batch: int = 64
    imgsz: int = 640
    data: OpenSeedDatasets = "OpenSeed-LZU-detect.yaml"  # "OpenSeed-LZU-detect.yaml", "OpenSeed-LZU-seg.yaml"
    seed: int = 42
    repeat_mode: bool = False
    seeds: list[int] = [0, 21, 42, 2541, 3407]
    # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    optimizer: Optimizers = "SGD"

    pretrained: bool = False
    freeze: list[int] = []
    resume: bool = False
    device: str = "0"
    cache: bool = False  # False disk ram
    workers: int = 8
    exist_ok: bool = False
    verbose: bool = True
    deterministic: bool = True
    amp: bool = True
    compile: bool = False

    # Hyperparameters
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    copy_paste: float = 0.0
    copy_paste_mode: Literal["flip", "mixup"] = "flip"

    def process_args(self):
        return
        if self.task == "segment" and self.model.endswith(".yaml") and "seg" not in self.model:
            warnings.warn(f"`segment` is given but model is `{self.model}`, switching to segmentation model")
            self.model = self.model.replace(".yaml", "-seg.yaml")
        if self.task == "detect" and self.model.endswith(".yaml") and "seg" in self.model:
            warnings.warn(f"`detect` is given but model is `{self.model}`, switching to detection model")
            self.model = self.model.replace("-seg.yaml", ".yaml")
        if self.repeat_mode:
            assert len(self.seeds) > 0, "In repeat_mode, seeds list must be non-empty"

        if self.task == "detect" and "detect" not in self.data:
            warnings.warn(f"`detect` is given but data is `{self.data}`, switching to detection data")
            self.data = "OpenSeed-LZU-detect.yaml"
        if self.task == "segment" and "seg" not in self.data:
            warnings.warn(f"`segment` is given but data is `{self.data}`, switching to segmentation data")
            self.data = "OpenSeed-LZU-seg.yaml"


def train_one_seed(args: OpenSeedArgs, seed: int):
    args.seed = seed
    trainer = None
    if "yoloe" in args.model:
        # we do not train YOLOE from scratch
        model = YOLOE(args.model, task=args.task, verbose=args.verbose)
        trainer = YOLOEPETrainer if args.task == "detect" else YOLOEPESegTrainer
        assert args.model in MODEL_WEIGHTS, f"Model {args.model} not supported"
        model.load(MODEL_WEIGHTS[args.model])
    elif "worldv2" in args.model:
        # we do not train YOLO World from scratch
        model = YOLOWorld(args.model, task=args.task, verbose=args.verbose)
        assert args.model in MODEL_WEIGHTS, f"Model {args.model} not supported"
        model.load(MODEL_WEIGHTS[args.model])
    elif args.model.startswith("yolo"):
        model = YOLO(args.model, task=args.task, verbose=args.verbose)
        if args.pretrained:
            assert args.model in MODEL_WEIGHTS, f"Model {args.model} not supported"
            model.load(MODEL_WEIGHTS[args.model])
    elif args.model.startswith("rtdetr"):
        model = RTDETR(args.model)
        if args.pretrained:
            assert args.model in MODEL_WEIGHTS, f"Model {args.model} not supported"
            model.load(MODEL_WEIGHTS[args.model])
    else:
        raise ValueError(f"Model {args.model} not supported")

    name = (
        f"{args.model.split('.')[0]}-"
        f"{args.task}-"
        f"e{args.epochs}-"
        f"b{args.batch}-"
        f"sz{args.imgsz}-"
        f"cp{args.copy_paste}-"
        f"opt{args.optimizer}-"
        f"s{args.seed}-"
    )
    name += "pre-" if args.pretrained else ""

    save_dir = Path(RUNS_DIR) / args.task / name
    if (save_dir / "results.csv").exists() and not args.exist_ok:
        warnings.warn(
            f"Results for {name} already exist at {save_dir}, skipping. Use 'exist_ok=True' to overwrite."
        )
        return None, None

    # Train the model
    results = model.train(
        name=name,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        resume=args.resume,
        optimizer=args.optimizer,
        pretrained=args.pretrained,
        freeze=args.freeze,
        device=args.device,
        cache=args.cache,
        workers=args.workers,
        exist_ok=args.exist_ok,
        verbose=args.verbose,
        deterministic=args.deterministic,
        amp=args.amp,
        compile=args.compile,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        copy_paste=args.copy_paste,  # S:0.15; L:0.5; X:0.6
        copy_paste_mode=args.copy_paste_mode,
        scale=0.5,  # S:0.9; L:0.9; X:0.9
        mosaic=1.0,
        mixup=0.0,  # S:0.05; L:0.15; X:0.2
        trainer=trainer,
    )


def main(args: OpenSeedArgs):
    if args.repeat_mode:
        for seed in args.seeds:
            print(f"\n--- Training with seed {seed} ---")
            train_one_seed(args, seed)
    else:
        train_one_seed(args, args.seed)


if __name__ == "__main__":
    args = OpenSeedArgs().parse_args()
    # args = OpenSeedArgs().parse_args("--task detect --model yolov13n.yaml --data OpenSeed-LZU-detect.yaml --batch 16 --device 'cpu' --workers 14 --resume --exist_ok".split())
    # args = OpenSeedArgs().parse_args("--task segment --model yolo11n-seg.yaml --data OpenSeed-LZU-seg.yaml --batch 16 --device 'cpu' --workers 14 --resume --exist_ok".split())
    print(args)
    main(args)
