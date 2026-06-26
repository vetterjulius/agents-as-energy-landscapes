from abc import ABC, abstractmethod

class EnergyTerm(ABC):

    def __init__(self, weight=1.0):
        self.weight = weight

    @abstractmethod
    def compute(self, state):
        pass

    @property
    def name(self):
        return self.__class__.__name__
