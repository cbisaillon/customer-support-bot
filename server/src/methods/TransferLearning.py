from .method import Method
from transformers import GPT2DoubleHeadsModel, GPT2Tokenizer
from itertools import chain

# Let's define our contexts and special tokens
persona = [["i", "like", "playing", "football", "."],
           ["i", "am", "from", "NYC", "."]]
history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]
reply = ["great", "to", "hear"]
bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"


class TransferLearning(Method):

    def __init__(self, context):
        super(TransferLearning, self).__init__(context)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.model = GPT2DoubleHeadsModel.from_pretrained('gpt2-medium')

        # ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
        # ``additional_special_tokens``
        self.special_tokens = {
            'bos_token': "<bos>",
            'eos_token': "<eos>",
            'additional_special_tokens': ["<speaker1>", "<speaker2>"],
            'pad_token': "<pad>"
        }

        self.tokenizer.add_special_tokens(self.special_tokens)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)

    def build_inputs(self, persona, history, reply):
        sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
        sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                    enumerate(sequence[1:])]

        words = list(chain(*sequence))
        segments = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        position = list(range(len(words)))
        return words, segments, position, sequence

    def respond(self, question):
        words, segments, position, sequence = self.build_inputs(persona, history, reply)

        print(sequence)

        words = self.tokenizer.convert_tokens_to_ids(words)
        segments = self.tokenizer.convert_tokens_to_ids(segments)

        return ""
