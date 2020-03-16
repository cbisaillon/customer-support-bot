from flask import Flask, json

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
tokenizer.save_pretrained("dataset/DialoGPT-large/")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")

api = Flask(__name__)

@api.route('/', methods=['GET'])
def getResponse():
    text = tokenizer.encode("Hello" + tokenizer.eos_token, return_tensors='pt')
    response = model.generate(text, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(response[:, text.shape[-1]:][0], skip_special_tokens=True)


if __name__ == '__main__':
    api.run()
