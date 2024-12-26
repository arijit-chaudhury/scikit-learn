from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
print(X)
#print(y)

model = KNeighborsRegressor().fit(X, y)
predict_data = model.predict(X)
print(predict_data)