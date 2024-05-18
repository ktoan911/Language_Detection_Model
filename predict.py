import re
from transformers import TextClassificationPipeline

def predcit(model, tokenizer, text_list, id2label):
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0)

    predict = pipe(text_list)
    result = []
    for i in predict:
        language_num = re.findall(r'\d+', i['label'])
        result.append(id2label[int(language_num[0])])
    return result
