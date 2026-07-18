# scripts/train.py: train a method (default: reg) using Trainer

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import parse_args, get_output_dir, get_wrapper_kwargs
from src.utils.io import save_model
from src.core.factory import get_dataloader, get_wrapper
from src.core.trainer import Trainer


def main():
    args = parse_args()
    output_dir = args.output_dir or get_output_dir(args)

    train_loader = get_dataloader("train", args.csv_path, image_size=args.image_size,
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
        num_samples=args.train_size)
    valid_loader = get_dataloader("valid", args.csv_path, image_size=args.image_size,
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
        num_samples=args.valid_size)

    wrapper = get_wrapper(args.method, device=args.device, **get_wrapper_kwargs(args))
    trainer = Trainer(wrapper, output_dir=output_dir)
    history = trainer.fit_early_stop(train_loader, valid_loader,
        max_epochs=args.max_epochs, patience=args.patience)
    trainer.save(history)

    if args.save:
        checkpoint = args.checkpoint or os.path.join(output_dir, "model.pth")
        save_model(wrapper.model, checkpoint)
        trainer.logger.info("model saved to %s" % checkpoint)


if __name__ == "__main__":
    main()
