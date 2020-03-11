import torch
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from operator import itemgetter
from nltk.tokenize import sent_tokenize

EOS = 0
SOS = 1
UNKNOWN = 2


class Dictionary:

    def __init__(self, max_sentence_length):
        self.max_sentence_length = max_sentence_length

        self.word2Index = {"<SOS>": SOS, "<EOS>": EOS, "<UNK>": UNKNOWN}
        self.index2Word = ["<SOS>", "<EOS>", "<UNK>"]
        self.word2Count = OrderedDict({"<SOS>": 1, "<EOS>": 1, "<UNK>": 1})
        self.nbWords = 3

    def addWord(self, word):
        if not word in self.word2Index.keys():
            self.word2Index[word] = len(self.word2Index.keys())
            self.index2Word.append(word)
            self.word2Count[word] = 1
            self.nbWords += 1
        else:
            self.word2Count[word] += 1


    def addSentence(self, sentence):
        parsedSentence = self.parseSentence(sentence)
        for word in parsedSentence:
            self.addWord(word)

    def parseSentence(self, sentence, remove_unknown_words=False):
        if type(sentence) is list:
            sentence = ' '.join(sentence)

        # Remove star character
        sentence = sentence.replace("*", "")

        sentence = word_tokenize(sentence.lower())
        sentence = ["<SOS>"] + sentence

        # Cut the sentence to maximum length
        sentence = sentence[:self.max_sentence_length - 1] # Minus one for the EOS token

        sentence.append("<EOS>")

        if remove_unknown_words:
            # Remove unknown words
            # Todo: Maybe not good for performance
            sentence = [word for word in sentence if word in self.word2Index.keys()]

        return sentence

    def keepNWords(self, n):
        """
        Remove the least used words until the vocabulary size is of size n
        :param n: the size that the vocabulary should be
        """
        print(self.word2Count)

    def oneHotEncode(self, word):
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
