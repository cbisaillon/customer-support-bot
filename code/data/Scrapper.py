import os
import praw
import pprint

subreddits = [
    "askreddit",
    # "askphilosophy",
    # 'askscience',
    # 'askengineers',
    # 'askculinary'
]


class Scrapper:
    def __init__(self):
        self.secret_key = os.getenv("REDDIT_SECRET")
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.user_agent = os.getenv("REDDIT_USER_AGENT")

        self.reddit = praw.Reddit(client_id=self.client_id,
                                  client_secret=self.secret_key,
                                  user_agent=self.user_agent)

        print(self.reddit.read_only)

    def getPosts(self, limit):
        for subreddit in subreddits:
            for submission in self.reddit.subreddit(subreddit).hot(limit=limit):
                # print(submission.title)
                # print(len(submission.comments))
                replies = self.getReplies(submission)
                if replies is not None:
                    for reply in replies:
                        print()
                        print(reply[0])
                        print(reply[1])

    def processComment(self, comment):
        #todo
        return comment.split()

    def getReplies(self, post):
        print("A")
        comment_with_response = []

        if len(post.comments) > 0:
            comments = post.comments.list()
        else:
            print("Post has no comments. Ignoring")
            return

        for comment in comments:
            if len(comment._replies) > 0:
                for sub_comment in comment.replies.list():
                    if hasattr(sub_comment, 'body'):
                        comment_with_response.append((self.processComment(comment.body), self.processComment(sub_comment.body)))

                return comment_with_response

        else:
            print("Comment has no replies. Ignoring")



