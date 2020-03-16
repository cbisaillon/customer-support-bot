from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from flask import Flask, json, render_template, request

api = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

context = r"""
Random Company Name (RCN) is a company built in 2014 by Carole Shaw and Alfred Campbell. 
The aim of our company is to deliver effective and easy to use natural language processing software
to be used by anyone. We are open Monday through friday from 9am to 16pm.
Random Company Name is not open Saturday and Sunday (We are not open on the weekends). We do not have an online store. 
You can come visit our office at 172 East Piper Burnsville, MN 55337.
We can help you get to our office by clicking this link https://www.google.ca/map.
Our phone number is 514-222-4444.
"""


# questions = [
#     "What are you doing?",
#     "What is the name of your company?",
#     "Who created this business?",
#     "When did you create this business?",
#     "When are you open?"
#     "Are you open on the weekends?",
#     "When do you open on Tuesdays?",
#     "When do you open on Saturday?",
#     "Where are you located?",
#     "How can I call you?",
#     "What is your phone number?",
#     "Help me get to your office",
#     "How can I get to your office?",
#     "Are you a robot?"
# ]

def respond(question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start: answer_end]))
    return answer


@api.route('/')
def askQuestion():
    question = request.args.get('question')

    return respond(question)


if __name__ == '__main__':
    api.run(debug=True, host="127.0.0.1", port=8080)
