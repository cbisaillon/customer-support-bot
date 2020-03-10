from dotenv import load_dotenv, find_dotenv
from code.data.Scrapper import Scrapper
from code.dictionary import Dictionary
from code.networks.networks import ChatBot
import pandas as pd
import getopt
import sys
from nltk.corpus import brown
import torch
import torch.nn as nn
import time
import math

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


def buildCorpus(redditData):
    print("Building corpus...")
    for sentence in brown.sents():
        dictionary.addSentence(dictionary.parseSentence(sentence))

    print("Brown corpus built.")
    print("Adding reddit data to dictionary")

    for comment in redditData['comment']:
        dictionary.addSentence(comment)

    for reply in redditData['reply']:
        dictionary.addSentence(reply)

    print("Built corpus")


criterion = nn.CrossEntropyLoss()
lr = 5.0
bptt = 35
print_interval = 200
train_batch_size = 20
eval_batch_size = 10

best_val_loss = float("inf")
epochs = 3
best_model = None


def get_batch(data, index):
    # todo
    pass


def train(model, optimizer, scheduler, data, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = dictionary.nbWords

    # todo: data have to be a tensor
    for batch, i in enumerate(range(0, data.size(0) - 1, bptt)):
        data, targets = get_batch(data, i)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % print_interval == 0 and batch > 0:
            cur_loss = total_loss / print_interval
            elapsed = time.time() - start_time

            print("Epoch {:3d} | {:5d}/{:5d} batches | loss: {:5.2f}".format(epoch, batch, train_batch_size, cur_loss))
            total_loss = 0
            start_time = time.time()


def evaluate(model, data_source):
    model.eval()
    total_loss = 0
    ntokens = dictionary.nbWords
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets)

    return total_loss / (len(data_source) - 1)


def splitTrainAndTestData(data):
    """
    Splits the data into training data and test data
    70% goes in train data and 30% in test
    :param data: the data to split
    :return: a tuple: (train_data, test_data)
    """
    nb_data = len(data)
    split_idx = math.floor(0.70 * nb_data)

    return data[:split_idx], data[split_idx:]


def convertDataToTensors(train_data, test_data):
    print("Converting train and test data to tensors")
    train_data.apply(lambda x: dictionary.entryToTensor(x), axis=1)
    test_data.apply(lambda x: dictionary.entryToTensor(x), axis=1)
    print(train_data.head())


def main():
    # Read the comments data
    data = pd.read_pickle(pickle_file)
    buildCorpus(data)
    print(dictionary.nbWords)
    exit()
    # Hyper parameters
    ntokens = dictionary.nbWords
    embedind_dimension = 200
    number_hidden = 200
    number_layers = 2
    number_attention_head = 2
    dropout = 0.2

    chatbot = ChatBot(ntokens, embedind_dimension, number_attention_head, number_hidden, number_layers, dropout)

    optimizer = torch.optim.SGD(chatbot.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    train_data, test_data = splitTrainAndTestData(data)
    train_tensor, test_tensor = convertDataToTensors(train_data, test_data)

    # Train the model
    print("Training model...")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(chatbot, optimizer, scheduler, data=train_data)
        val_loss = evaluate(chatbot, test_data)

        print("-" * 89)
        print(
            "| End of epoch {:3d} | time: {5.2f}s | valid loss {:5.2f}".format(epoch, (time.time() - epoch_start_time),
                                                                               val_loss))
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = chatbot

            # Save the model
            print("Better model found. Saving...")
            torch.save(chatbot.state_dict(), "network.pt")
            print("Saving done.")

        scheduler.step()


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
