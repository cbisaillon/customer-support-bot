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

device = torch.device("cpu")
if torch.cuda.is_available():
    print("USING CUDA")
    device = torch.device("cuda:0")


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
        dictionary.addSentence(sentence)

    # print("Brown corpus built.")
    # print("Adding reddit data to dictionary")

    # for comment in redditData['comment']:
    #     dictionary.addSentence(comment)
    #
    # for reply in redditData['reply']:
    #     dictionary.addSentence(reply)

    print("Built corpus")


criterion = nn.CrossEntropyLoss().to(device)
lr = 5.0
bptt = 35
print_interval = 200
train_batch_size = 20
eval_batch_size = 10
batch_size = 20
max_sentence_length = 15
vocab_size = 2000

best_val_loss = float("inf")
epochs = 3
best_model = None

dictionary = Dictionary(max_sentence_length)


def sentenceToTensor(sentence):
    """
    Transforms a sentence to a tensor of size (sentence size, word in dictionary)
    :param sentence: the sentence to make into a tensor
    :return: a tensor with each row corresponding to the one hot encoding of the word
    """

    parsed_sentence = dictionary.parseSentence(sentence, remove_unknown_words=True)
    tensor = torch.zeros((max_sentence_length, dictionary.nbWords), dtype=torch.long)

    for i, word in enumerate(parsed_sentence):
        one_hot_word = dictionary.oneHotEncode(word)
        tensor[i] = one_hot_word

    return tensor


def batchToTensor(batch):
    comment_tensors = torch.zeros((batch_size, max_sentence_length, dictionary.nbWords * 2), dtype=torch.long)
    # reply_tensor = torch.zeros((batch_size, max_sentence_length, dictionary.nbWords), dtype=torch.long)

    i = 0
    for index, entry in batch.iterrows():
        comment = entry['comment']
        reply = entry['reply']

        comment_tensor = sentenceToTensor(comment)
        reply_tensor = sentenceToTensor(reply)

        comment_tensors[i] = torch.cat([comment_tensor, reply_tensor], dim=1)
        i += 1

    return comment_tensors.to(device)


def get_batch(tensor, index):
    comment = torch.narrow(tensor[index], 1, 0, dictionary.nbWords)
    reply = torch.narrow(tensor[index], 1, dictionary.nbWords, dictionary.nbWords)
    return comment, reply


def train(model, optimizer, scheduler, data_batched, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = dictionary.nbWords

    for batchIdx, batch in enumerate(data_batched):
        batchTensor = batchToTensor(batch)

        # Go through each comment/reply in the batch
        for batch, i in enumerate(range(0, batchTensor.size(0) - 1)):
            comment, reply = get_batch(batchTensor, i)

            optimizer.zero_grad()
            output = model(comment)
            loss = criterion(output, reply)
            print(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % print_interval == 0 and batch > 0:
                cur_loss = total_loss / print_interval
                elapsed = time.time() - start_time

                print("Epoch {:3d} | {:5d}/{:5d} batches | loss: {:5.2f}".format(epoch, batchIdx, train_batch_size,
                                                                                 cur_loss))
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

    return convertDataToTensors(data[:split_idx], data[split_idx:], batch_size)


def convertDataToTensors(train_data, test_data, max_size_per_batch):
    print("Converting train and test data to tensors")

    # Split the data into multiple dataframe
    train_dataframe_size = math.ceil(len(train_data) / max_size_per_batch)
    test_dataframe_size = math.ceil(len(test_data) / max_size_per_batch)

    train_dataframes = []
    for i in range(train_dataframe_size):
        train_batch = train_data[max_size_per_batch * i: max_size_per_batch * (i + 1)]
        train_dataframes.append(train_batch)

    test_dataframes = []
    for i in range(test_dataframe_size):
        test_batch = test_data[max_size_per_batch * i: max_size_per_batch * (i + 1)]
        test_dataframes.append(test_batch)

    return train_dataframes, test_dataframes


def main():
    # Read the comments data
    data = pd.read_pickle(pickle_file)
    buildCorpus(data)

    # Hyper parameters
    embedind_dimension = 20
    number_hidden = 1
    number_layers = 1
    number_attention_head = 1
    dropout = 0.2

    chatbot = ChatBot(vocab_size, embedind_dimension, number_attention_head, number_hidden, number_layers, dropout).to(device)

    optimizer = torch.optim.SGD(chatbot.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    train_data_batched, test_data_batched = splitTrainAndTestData(data)

    # Train the model
    print("Training model...")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(chatbot, optimizer, scheduler, data_batched=train_data_batched, epoch=epoch)
        val_loss = evaluate(chatbot, test_data_batched)

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
