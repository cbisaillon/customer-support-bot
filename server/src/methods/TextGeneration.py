from .method import Method
from transformers import BertTokenizer, PreTrainedEncoderDecoder
import torch
import torch.nn as nn


# class MessageGenerator(nn.Module):
#
#     def __init__(self):
#         super(MessageGenerator, self).__init__()
#
#         self.model = AutoModelWithLMHead.from_pretrained("distilbert-base-uncased-distilled-squad")
#
#
#     def forward(self, question, hidden_state):
#         pass

class FixedPreTrainedEncoderDecoder(PreTrainedEncoderDecoder):

    @staticmethod
    def prepare_model_kwargs(**kwargs):
        """ Prepare the encoder and decoder's keyword arguments.
        Keyword arguments come in 3 flavors:
        - encoder-specific (prefixed by `encoder_`)
        - decoder-specific (prefixed by `decoder_`)
        - those that apply to the model as whole.
        We let the specific kwargs override the common ones in case of
        conflict.
        """
        kwargs_common = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("encoder_") and not argument.startswith("decoder_")
        }
        decoder_kwargs = kwargs_common.copy()
        encoder_kwargs = kwargs_common.copy()
        encoder_kwargs.update(
            {
                argument[len("encoder_"):]: value
                for argument, value in kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        decoder_kwargs.update(
            {
                argument[len("decoder_"):]: value
                for argument, value in kwargs.items()
                if argument.startswith("decoder_")
            }
        )
        decoder_kwargs["encoder_attention_mask"] = encoder_kwargs.get("attention_mask", None)

        return encoder_kwargs, decoder_kwargs


class TextGeneration(Method):

    def __init__(self, context):
        super(TextGeneration, self).__init__(context)

        self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        self.model = FixedPreTrainedEncoderDecoder.from_pretrained("bert-large-cased", "bert-large-cased")
        self.pastQuestions = []

    def respond(self, question):
        self.pastQuestions.append(question)

        encoded_question = self.tokenizer.encode(question)
        question_tensor = torch.tensor([encoded_question])

        answer = "[CLS]"
        encoded_answer = self.tokenizer.encode(answer, add_special_tokens=False)

        answer_tensor = torch.tensor([encoded_answer], dtype=torch.long)

        self.model.eval()

        question_tensor = question_tensor.to('cuda')
        answer_tensor = answer_tensor.to('cuda')
        self.model.to('cuda')

        max_length = 10

        with torch.no_grad():

            hidden_state = None

            for i in range(max_length):

                params = {
                    "hidden_states": hidden_state
                }

                outputs = self.model(question_tensor, answer_tensor, params)

                predictions = outputs[0]
                hidden_state = outputs[1]

                predicted_index = torch.tensor([torch.argmax(predictions[0, -1])]).to('cuda')

                print(answer_tensor)
                # Append the new word to the answer
                answer_tensor[0] = torch.cat((answer_tensor[0], predicted_index))

        print(answer_tensor)

        predicted_tokens = self.tokenizer.convert_ids_to_tokens(answer_tensor)

        return predicted_tokens
