import os
import praw
import pprint
import pandas as pd
from prawcore.exceptions import ServerError

subreddits = {
    "askreddit",
    "askphilosophy",
    'askscience',
    'askengineers',
    'askculinary',
    'worldnews',
    'movies',
    'news',
    'showerthoughts',
    'explainlikeimfive',
    'books',
    'story',
    'philosophy',
    'politics',
    # 'todayilearned',
    'aww',
    'music',
    'blog',
    'gifs',
    'television',
    'lifeprotips',
    'space',
    'diy',
    'gadgets',
    'food',
}


class Scrapper:
    def __init__(self):
        self.secret_key = os.getenv("REDDIT_SECRET")
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.user_agent = os.getenv("REDDIT_USER_AGENT")

        self.reddit = praw.Reddit(client_id=self.client_id,
                                  client_secret=self.secret_key,
                                  user_agent=self.user_agent)

    def createDataSet(self, post_limit, comment_limit_per_post, save_file):
        dataset = pd.DataFrame({'comment': [], 'reply': []})
        for subreddit in subreddits:
            print("doing: {}".format(subreddit))
            for submission in self.reddit.subreddit(subreddit).hot(limit=post_limit):
                # print(submission.title)
                # print(len(submission.comments))
                try:
                    replies = self.getReplies(submission, comment_limit_per_post)
                    if replies is not None:
                        dataset = dataset.append(pd.DataFrame(replies, columns=dataset.columns))
                except Exception as e:
                    print("Server error: {}".format(str(e)))



        # Save the dataset
        dataset.to_pickle(save_file)

        return dataset

    def processComment(self, comment):
        # todo
        return comment.split()

    def getReplies(self, post, comment_limit):
        comment_with_response = []

        if len(post.comments) > 0:
            comments = post.comments.list()
        else:
            # print("Post has no comments. Ignoring")
            return

        for index, comment in enumerate(comments):
            if hasattr(comment, '_replies'):
                if len(comment._replies) > 0:
                    for sub_comment in comment.replies.list():
                        if hasattr(sub_comment, 'body'):
                            comment_with_response.append((comment.body, sub_comment.body))

            if index > comment_limit:
                break

        return comment_with_response
