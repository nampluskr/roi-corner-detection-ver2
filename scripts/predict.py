# scripts/predict.py: run a checkpoint on the test split and save predictions.csv

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import get_output_dir, get_wrapper_kwargs, parse_args
from src.core.factory import get_dataloader, get_wrapper
from src.core.predictor import Predictor
from src.utils.io import load_model


def main():
    """Load one checkpoint, predict its test split, and save corner CSV rows."""
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for prediction")
    output_dir = args.output_dir or get_output_dir(args)
    test_loader = get_dataloader(
        "test", args.csv_path, image_size=args.image_size, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers, num_samples=args.test_size)
    wrapper = get_wrapper(args.method, device=args.device, **get_wrapper_kwargs(args))
    load_model(wrapper.model, args.checkpoint)
    predictor = Predictor(wrapper, output_dir=output_dir)
    rows = predictor.predict(test_loader)
    path = predictor.save(rows)
    print("saved predictions to %s" % path)


if __name__ == "__main__":
    main()
