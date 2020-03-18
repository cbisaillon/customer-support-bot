from .method import Method
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch


class SimpleAnswerExtraction(Method):

    def __init__(self, context):
        super(SimpleAnswerExtraction, self).__init__(context)

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    def respond(self, question):
        # input_ids = torch.tensor(tokenizer.encode(question, context)).unsqueeze(0)
        inputs = self.tokenizer.encode_plus(question, self.context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        answer_start_scores, answer_end_scores = self.model(input_ids)
        # print(model(input_ids))

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[0][answer_start.data: answer_end.data].tolist()))
        return answer
