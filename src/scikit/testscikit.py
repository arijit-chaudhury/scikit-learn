from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer


class Data:
    def __init__(self):
        print('Init')

    def learn_data(self):
        X, y = load_breast_cancer(return_X_y=True)
        print(X)
        model = KNeighborsRegressor().fit(X, y)
        predict_data = model.predict(X)
        print(predict_data)


if __name__ == '__main__':
    Data().learn_data()
