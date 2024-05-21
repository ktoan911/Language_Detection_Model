import os
import numpy as np
from argparse import ArgumentParser
import model
import trainer
import data
from datasets import load_metric

if __name__ == '__main__':
    parser = ArgumentParser()

    # Khai báo tham số cần thiết
    # lora
    parser.add_argument("--lora_alpha", default=30, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--lora_r", default=30, type=int)

    # link to save model
    parser.add_argument("--output-dir", default='./results', type=str)

    # training param
    parser.add_argument("--num-train-epochs", default=3, type=int)
    parser.add_argument("--per-device-train_batch_size", default=16, type=int)
    parser.add_argument("--per-device-eval-batch-size", default=16, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--optim", default="paged_adamw_32bit", type=str)
    parser.add_argument("--save-steps", default=0, type=int)
    parser.add_argument("--logging-steps", default=25, type=int)
    parser.add_argument("--learning-rate", default=30, type=int)
    parser.add_argument("--weight-decay", default=0.001, type=float)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--bf16", default=False, type=bool)
    parser.add_argument("--max-grad-norm", default=0.3, type=float)
    parser.add_argument("--max-steps", default=-1, type=int)
    parser.add_argument("--warmup-ratio", default=0.03, type=int)
    parser.add_argument("--group-by-length", default=True, type=bool)
    parser.add_argument("--lr-scheduler-type", default="cosine", type=str)
    parser.add_argument("--max-seq-length", default=256, type=int)

    # loading model param
    parser.add_argument("--model-name", required=True, type=str)

    # loading data param
    parser.add_argument("--train-path", required=True, type=str)
    parser.add_argument("--test-path", required=True, type=str)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # Khởi tạo dữ liệu
    data_generator = data.Data(model_name=args.model_name,
                               train_path=args.train_path, test_path=args.test_path)

    train_dataset, test_dataset, label2id, id2label = data_generator.process_dataset()

    # Khởi tạo mô hình
    model_generator = model.Model(
        base_model_name=args.model_name, num_labels=len(label2id))
    model = model_generator.get_peft_model(
        label2id, id2label, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, lora_r=args.lora_r, bias='none')

    # accuracy
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Khởi tạo trainer
    train_generator = trainer.Train()
    training_args = train_generator.get_training_args(output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
                                                      per_device_train_batch_size=args.per_device_train_batch_size,
                                                      per_device_eval_batch_size=args.per_device_eval_batch_size,
                                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                      optim=args.optim, save_steps=args.save_steps, logging_steps=args.logging_steps,
                                                      learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                                      fp16=args.fp16, bf16=args.bf16, max_grad_norm=args.max_grad_norm,
                                                      max_steps=args.max_steps, warmup_ratio=args.warmup_ratio,
                                                      group_by_length=args.group_by_length, lr_scheduler_type=args.lr_scheduler_type)

    train = train_generator.fit(model=model, training_args=training_args, train_dataset=train_dataset,
                                test_dataset=test_dataset, compute_metrics=compute_metrics)
