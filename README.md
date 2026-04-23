# OpenSeed-detect

End-to-End Grass Seed Mixture Detection with a Synthetic Dataset

## Getting started

1. install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. clone this repo
```bash
git clone https://github.com/HuLab-LZU/OpenSeed-detect.git OpenSeed-detect
cd OpenSeed-detect
```
3. install dependencies
```bash
uv sync
```
4. activate virtual environment
```bash
source .venv/bin/activate
```
5. training
```bash
python train.py --task detect --model yolo-master-x.yaml --data OpenSeed-LZU-detect.yaml --batch 16 --device 0 --workers 8 --epochs 100 --resume --repeat_mode --exist_ok --seeds 0 21 42 2541 3407
```
explore more options from the scripts or using `python train.py --help`

## Cite

TODO: add citation info.
