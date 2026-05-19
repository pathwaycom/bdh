"""Run BDH training, evaluation, and memory demos in order."""

import argparse

from bdh_app.training import run_training
from bdh_app.evaluation import run_evaluation, run_multi_prompt_evaluation
from bdh_app.memory import run_memory_demo


def main():
    parser = argparse.ArgumentParser(description="BDH end-to-end runner")
    parser.add_argument("--max_iters", type=int, default=3000, help="Training iterations")
    parser.add_argument("--batch_size", type=int, default=12, help="Training batch size")
    parser.add_argument("--base_out_dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--prompt", type=str, default="To be or not to be", help="Evaluation prompt")
    args = parser.parse_args()

    train_out_dir = f"{args.base_out_dir}/training"
    eval_out_dir = f"{args.base_out_dir}/evaluation"
    memory_out_dir = f"{args.base_out_dir}/memory"

    run_training(args.max_iters, args.batch_size, train_out_dir)
    run_evaluation(
        input_dir=f"{train_out_dir}/checkpoints",
        output_dir=eval_out_dir,
        prompt=args.prompt,
        device="auto",
        log_file=None,
    )
    run_multi_prompt_evaluation(
        input_dir=f"{train_out_dir}/checkpoints",
        output_dir=eval_out_dir,
        prompts=[
            "My lord, I shall",
            "The king hath",
            "Upon this day",
        ],
        device="auto",
    )
    run_memory_demo(train_out_dir=train_out_dir, out_dir=memory_out_dir)


if __name__ == "__main__":
    main()
