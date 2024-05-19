from transformers import (TrainingArguments, Trainer)


class Train:
    def get_training_args(self, output_dir, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size,
                          gradient_accumulation_steps, optim, save_steps, logging_steps, learning_rate, weight_decay,
                          fp16, bf16, max_grad_norm, max_steps, warmup_ratio, group_by_length, lr_scheduler_type):
        return TrainingArguments(
            output_dir=output_dir,    # thư mục lưu model
            num_train_epochs=num_train_epochs,  # số lần lặp
            per_device_train_batch_size=per_device_train_batch_size,  # độ lớn mỗi batch
            per_device_eval_batch_size=per_device_eval_batch_size,
            # độ tích lũy gradient descent trước khi cập nhập trọng số
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            # số bước mà sau đó các thông số về lost, accuracy được trả về
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,  # nhóm các dữ liệu với nhau dựa trên độ dài
            lr_scheduler_type=lr_scheduler_type,
            report_to="tensorboard",
            remove_unused_columns=False
        )

    def fit(self, model, train_dataset, test_dataset, training_args, compute_metrics):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()
        return trainer
