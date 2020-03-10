import torch
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

SOS = 0
EOS = 1
UNKNOWN = 2


class Dictionary:

    def __init__(self):
        self.word2Index = {"<SOS>": SOS, "<EOS>": EOS, "<UNK>": UNKNOWN}
        self.index2Word = ["<SOS>", "<EOS>", "<UNK>"]
        self.nbWords = 3

    def addWord(self, word):
        if not word in self.word2Index:
            self.word2Index[word] = len(self.word2Index.keys())
            self.index2Word.append(word)
            self.nbWords += 1

    def addSentence(self, sentence):
        for word in self.parseSentence(sentence):
            self.addWord(word)

    def parseSentence(self, sentence):
        if type(sentence) is list:
            sentence = ''.join(sentence)

        sentence = word_tokenize(sentence.lower())
        return sentence

    def oneHotEncode(self, word):
        word = word.lower()
        one_hot = torch.zeros(self.nbWords)
        if word in self.word2Index.keys():
            one_hot[self.word2Index[word]] = 1
        else:
            print("{} unkownn".format(word))
            one_hot[UNKNOWN] = 1

        return one_hot

    def entryToTensor(self, sentences):
        comment = sentences['comment']
        reply = sentences['reply']

        comment = list(map(lambda word: self.oneHotEncode(word), word_tokenize(comment.lower())))
        reply = list(map(lambda word: self.oneHotEncode(word), word_tokenize(reply.lower())))

        return comment, reply
