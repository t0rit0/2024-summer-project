from transformers import AutoTokenizer , AutoModelForSequenceClassification , Trainer
from datasets import Dataset , DatasetDict
#import pandas as pd
import numpy as np
import torch

if __name__ == "__main__":
    # 加载模型和分词器
    model1_path = 'classificaiton_model/saved_model'
    model2_path = 'classificaiton_model/saved_model_response_judge'
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
    model1 = AutoModelForSequenceClassification.from_pretrained(model1_path)

    tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
    model2 = AutoModelForSequenceClassification.from_pretrained(model2_path)


    # text = "Tell me a joke about the cat"
    text = input('[INPUT]: ')
    inputs = tokenizer1(text, return_tensors="pt")

    with torch.no_grad():
        logits = model1(**inputs).logits
        print(logits)
        # print(torch.argmax(logits, dim=-1))
    predicted_class_id = logits.argmax().item()
    print(f'model1: {model1.config.id2label[predicted_class_id]}')

    # text = "Tell me a joke about the cat"
    inputs = tokenizer2(text, return_tensors="pt")

    with torch.no_grad():
        logits = model2(**inputs).logits
        print(logits)
        # print(torch.argmax(logits, dim=-1))
    predicted_class_id = logits.argmax().item()
    print(f'model2: {model2.config.id2label[predicted_class_id]}')

    # # 进行推理
    # outputs = model(**inputs)

    # # 处理输出
    # logits = outputs.logits
    # # predictions = torch.argmax(logits, dim=-1)
    # print(logits)
