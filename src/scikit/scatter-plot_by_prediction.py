from testscikit import Data


class PredictPlot:
    def __init__(self):
        print('Init')

    @classmethod
    def plot(self):
        Data.learn_data(self)


PredictPlot().plot()
