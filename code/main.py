from dotenv import load_dotenv, find_dotenv
from code.data.Scrapper import Scrapper
from code.dictionary import Dictionary
from code.networks.networks import ChatBot
import pandas as pd
import getopt
import sys
from nltk.corpus import brown

pickle_file = "../dataset/comment-reply.pkl"
dictionary = Dictionary()

def createDataSet():
    """
    Scrape reddit comments and create a dataset
    """
    print("Scrapping reddit comments")
    scrapper = Scrapper()
    scrapper.createDataSet(post_limit=150,
                           comment_limit_per_post=50,
                           save_file=pickle_file)


def buildCorpus():
    print("Building corpus...")
    for sentence in brown.sents():
        dictionary.addSentence(dictionary.parseSentence(sentence))

def main():
    # Read the comments data
    data = pd.read_pickle(pickle_file)
    buildCorpus()

    chatbot = ChatBot()

    print("Done: {}".format(dictionary.nbWords))


def printUsage():
    print("main.py [-c]")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc", ["help", "create"])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", '--help'):
            printUsage()
            sys.exit()
        if opt in ("-c", "--create"):
            createDataSet()

    if len(opts) == 0:
        main()
