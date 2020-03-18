from abc import ABC, abstractmethod


class Method(ABC):

    def __init__(self, context):
        self.context = context

    """
    Abstract class that all of the methods I implement will inherit
    """
    @abstractmethod
    def respond(self, question):
        pass
