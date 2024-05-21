from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


class Model:
    def __init__(self, base_model_name, num_labels):
        self.model_name = base_model_name
        self.num_labels = num_labels

    def get_model(self, label2id, id2label):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True)
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        return model

    def peft_config(self, lora_alpha, lora_dropout, lora_r, bias='none'):
        return LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias=bias,
            task_type='CLASSIFICATION',
        )

    def get_peft_model(self, label2id, id2label, lora_alpha, lora_dropout, lora_r, bias='none'):
        model = get_peft_model(
            self.get_model(label2id, id2label),
            self.peft_config(lora_alpha, lora_dropout, lora_r, bias)
        )
        return model
