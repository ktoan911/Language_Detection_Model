import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    parser.add_argument("--output_dir", default=30, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--per_device_train_batch_size", default=16, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--optim", default="paged_adamw_32bit", type=str)
    parser.add_argument("--save_steps", default=0, type=int)
    parser.add_argument("--logging_steps", default=25, type=int)
    parser.add_argument("--learning_rate", default=30, type=int)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--bf16", default=False, type=bool)
    parser.add_argument("--max_grad_norm", default=0.3, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_ratio", default=30, type=int)
    parser.add_argument("--group_by_length", default=True, type=bool)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()
