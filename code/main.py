import os
from dotenv import load_dotenv, find_dotenv
from code.data.Scrapper import Scrapper


def main():
    scrapper = Scrapper()
    scrapper.getPosts(limit=2)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())

    main()
