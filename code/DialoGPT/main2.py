from os import path
import os
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
tokenizer.save_pretrained("dataset/DialoGPT-large/")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")

for step in range(5):
    user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=1) if step > 0 else user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    print("Bot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

