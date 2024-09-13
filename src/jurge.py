from transformers import AutoTokenizer , AutoModelForSequenceClassification , Trainer
from datasets import Dataset , DatasetDict
from prompt_template import SORRY_LIST
#import pandas as pd
import numpy as np
import torch

MODEL1_PATH = 'classificaiton_model/saved_model'
MODEL2_PATH = 'classificaiton_model/saved_model_response_judge'

class Finetuned_model_jurge():
    def __init__(self, model_path:list[str]) -> None:
        self.tokenizers = [AutoTokenizer.from_pretrained(path) for path in model_path]
        self.models = [AutoModelForSequenceClassification.from_pretrained(path) for path in model_path]
    
    def each_jurge(self, text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            return logits.argmax().item() == 0
        
    def jurge(self, text):
        res = True
        for tokenizer, model in zip(self.tokenizers, self.models):
            res &= self.each_jurge(text, model, tokenizer)
        return res
    
class Word_list_jurge():
    def __init__(self, word_list:list[str]=SORRY_LIST) -> None:
        self.word_list = word_list
    
    def jurge(self, text):
        for word in self.word_list:
            if word in text:
                return True
        return False



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
    predicted_class_id = logits.argmax().item()
    print(f'model2: {model2.config.id2label[predicted_class_id]}')

