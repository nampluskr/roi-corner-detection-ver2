# scripts/config.py: default training configuration and argument parsing

import os
import argparse

VER1_DATA_DIR = "/mnt/d/projects/nampluskr/00_review/260701_roi-corner-detection-ver1/data"

DEFAULTS = dict(
    data_dir=VER1_DATA_DIR,
    csv_path=[
        os.path.join(VER1_DATA_DIR, "smartdoc", "gt_corners.csv"),
        os.path.join(VER1_DATA_DIR, "midv2020", "gt_corners.csv"),
    ],
    seed=42,
    dataset="public",
    method="reg",
    backbone="custom",
    head="coord_gap",
    model=None,
    image_size=224,
    batch_size=4,
    max_epochs=5,
    patience=2,
    num_workers=4,
    train_size=2000,  # None: all train samples
    valid_size=1000,  # None: all valid samples
    test_size=1000,  # None: all test samples
)


def cfg_get(cfg, key, default=None):
    """Return one config value from a dict or argparse namespace."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_experiment(cfg):
    """Build an experiment name string from method, batch_size, max_epochs, backbone, and head."""
    method = cfg_get(cfg, "method", DEFAULTS["method"])
    batch_size = cfg_get(cfg, "batch_size", DEFAULTS["batch_size"])
    max_epochs = cfg_get(cfg, "max_epochs", DEFAULTS["max_epochs"])
    backbone = cfg_get(cfg, "backbone", DEFAULTS["backbone"]) or DEFAULTS["backbone"]
    head = cfg_get(cfg, "head", DEFAULTS["head"]) or DEFAULTS["head"]
    return "%s_bs%d_ep%d_%s_%s" % (method, batch_size, max_epochs, backbone, head)


def get_model_name(cfg):
    """Return a model name segment derived from model override or backbone."""
    model = cfg_get(cfg, "model", None)
    if model:
        return model
    backbone = cfg_get(cfg, "backbone", DEFAULTS["backbone"]) or DEFAULTS["backbone"]
    head = cfg_get(cfg, "head", DEFAULTS["head"]) or DEFAULTS["head"]
    return "%s_%s" % (backbone, head)


def get_output_dir(cfg, base="outputs"):
    """Return the outputs/{dataset}/{method}/{model}/{exp_name} directory for the given config."""
    dataset = cfg_get(cfg, "dataset", DEFAULTS["dataset"])
    method = cfg_get(cfg, "method", DEFAULTS["method"])
    return os.path.join(base, dataset, method, get_model_name(cfg), get_experiment(cfg))


def get_wrapper_kwargs(args):
    """Collect wrapper constructor kwargs from parsed args, passing through only set values."""
    kwargs = {}
    if getattr(args, "backbone", None):
        kwargs["backbone"] = args.backbone
    if getattr(args, "head", None):
        kwargs["head"] = args.head
    return kwargs


def parse_args():
    """Parse command-line arguments for scripts/train.py, defaulting to the reg method."""
    parser = argparse.ArgumentParser()
    parser.set_defaults(**DEFAULTS)

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--csv_path", type=str, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--head", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--valid_size", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()
