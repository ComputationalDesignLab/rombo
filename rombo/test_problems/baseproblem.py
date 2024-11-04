from abc import ABC, abstractmethod

class TestFunction(ABC):

    "Base class definition for test/benchmark problems"
    dimensions: int
    negate: bool = False

    @abstractmethod
    def function(self, x):
        "Defines the expression for the test function"

    @abstractmethod
    def evaluate(self, x):
        "Evaluates the function at the given x points"

    @abstractmethod
    def utility(self, y):
        "Evaluates the utility of a set of outputs"



