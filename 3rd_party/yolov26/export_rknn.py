import glob
import shutil as sh
from pathlib import Path

import regex
import rootutils
import yaml
from tqdm import tqdm

rootutils.setup_root(".", indicator=".project-root-yolo26", pythonpath=True, cwd=True)

from ultralytics.models import YOLO

LOG_DIR = Path("../../runs/detect")

OUT_DIR = Path("rknn_models")

MODEL_VARIANTS = [
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
]

# runs/detect/yolo11l-detect-e100-b16-sz640-cp0.0-optSGD-s21-
EXPERIMENT_DIR_REGEX = regex.compile(
    r"^(?P<model_variant>{})"
    r"-(?P<experiment_name>.+)"
    r"-s(?P<seed>\d+)-$".format("|".join(MODEL_VARIANTS))
)
OVERWRITE = False

experiment_dirs = glob.glob(str(LOG_DIR / "*"))
for d in tqdm(experiment_dirs):
    # d = experiment_dirs[0]
    match = EXPERIMENT_DIR_REGEX.match(Path(d).name)
    if match is None:
        continue
    match_groups = match.groupdict()
    seed = match_groups["seed"]
    model_variant = match_groups["model_variant"]
    if seed != "0":
        continue

    print(f"processing {d}")
    with open(Path(d) / "args.yaml") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    out_model_path = OUT_DIR / f"{model_variant}-{seed}.rknn"
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    if out_model_path.exists():
        continue

    model_path = Path(d) / "weights" / "best.pt"
    model = YOLO(model_path).eval()
    path = model.export(
        format="rknn",
        name="rk3588",
        imgsz=(640, 640),
        opset=17,
        batch=1,
        end2end=True,
    )

    sh.move(f"{path}/best-rk3588.rknn", out_model_path)
