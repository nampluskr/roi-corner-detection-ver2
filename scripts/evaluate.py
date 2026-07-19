# scripts/evaluate.py: evaluate a checkpoint on the test split and save metrics.json

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import get_output_dir, get_wrapper_kwargs, parse_args
from src.core.evaluator import Evaluator
from src.core.factory import get_dataloader, get_wrapper
from src.utils.io import load_model


def main():
    """Load one checkpoint, evaluate its test split, and save scalar metrics."""
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation")
    output_dir = args.output_dir or get_output_dir(args)
    test_loader = get_dataloader(
        "test", args.csv_path, image_size=args.image_size, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers, num_samples=args.test_size)
    wrapper = get_wrapper(args.method, device=args.device, **get_wrapper_kwargs(args))
    load_model(wrapper.model, args.checkpoint)
    evaluator = Evaluator(wrapper, output_dir=output_dir)
    results = evaluator.evaluate(test_loader)
    path = evaluator.save(results)
    print("saved metrics to %s" % path)
    print(results)


if __name__ == "__main__":
    main()
