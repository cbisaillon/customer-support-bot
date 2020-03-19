from flask import Flask, json, render_template, request
from methods.SimpleAnswerExtraction import SimpleAnswerExtraction
from methods.TextGeneration import TextGeneration
from methods.TransferLearning import TransferLearning

api = Flask(__name__)

context = r"""
Random Company Name (RCN) is a company built in 2014 by Carole Shaw and Alfred Campbell. 
The aim of our company is to deliver effective and easy to use natural language processing software
to be used by anyone. We are open Monday through friday from 9am to 16pm.
Random Company Name is not open Saturday and Sunday (We are not open on the weekends). We do not have an online store. 
You can come visit our office at 172 East Piper Burnsville, MN 55337.
We can help you get to our office by clicking this link https://www.google.ca/map.
Our phone number is 514-222-4444.
"""

method = TransferLearning(context)


@api.route('/')
def askQuestion():
    question = request.args.get('question')
    if question:
        return method.respond(question)
    else:
        return "Please ask a question..."


if __name__ == '__main__':

    while True:
        question = input("How can I help you?")
        print("------- {}".format(method.respond(question)))
