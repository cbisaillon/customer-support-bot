import os
from dotenv import load_dotenv, find_dotenv
from code.data.Scrapper import Scrapper
import pandas as pd


pickle_file = "../dataset/comment-reply.pkl"

def main():
    scrapper = Scrapper()
    scrapper.createDataSet(post_limit=100,
                           comment_limit_per_post=50,
                           save_file=pickle_file)

    data = pd.read_pickle(pickle_file)
    print(data.head())


if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())

    main()
