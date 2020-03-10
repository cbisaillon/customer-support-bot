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
        for word in sentence:
            self.addWord(word)

    def parseSentence(self, sentence):
        if type(sentence) is list:
            sentence = ''.join(sentence)

        return sentence.lower().split()
