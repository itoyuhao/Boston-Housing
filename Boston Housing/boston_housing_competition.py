"""
File: boston_housing_competition.py
Name: Kevin Chen
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientist!

Note: I've tried linear regression, polynomial (degree 2) with lasso or ridge regression and random forest!
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, metrics
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def main():
    # Read data
    boston = pd.read_csv('boston_housing/train.csv')


    # Data preprocessing
    np_data = np.array(boston)
    total_data = pd.DataFrame(np_data, columns=['id', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv'])
    # total_data.describe()

    X = total_data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
    y = total_data[['medv']]

    # print(X.head(5))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.09, random_state = 6)

    standardizer = preprocessing.MinMaxScaler()
    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.transform(X_test)

    # Linear regression without regularization
    # model = LinearRegression()
    #
    # classifier = model.fit(X_train, y_train)
    #
    # y_train_pred = classifier.predict(X_train)
    # y_test_pred = classifier.predict(X_test)
    #
    # print(metrics.mean_squared_error(y_train, y_train_pred)**0.5)
    # print(metrics.mean_squared_error(y_test, y_test_pred)**0.5)


    # Polynomial & Regularization(Ridge)
    # model_poly = PolynomialFeatures(2)
    # train_poly_X = model_poly.fit_transform(X_train)
    # test_poly_X = model_poly.fit_transform(X_test)
    #
    # model_ridge = Ridge(alpha = 3.8)
    # ridge_classifier = model_ridge.fit(train_poly_X, y_train)
    # train_pred_y = ridge_classifier.predict(train_poly_X)
    # test_pred_y = ridge_classifier.predict(test_poly_X)
    #
    # print(metrics.mean_squared_error(y_train, train_pred_y)**0.5)
    # print(metrics.mean_squared_error(y_test, test_pred_y)**0.5)


    # # Polynomial & Regularization(Lasso)
    # model_poly = PolynomialFeatures(degree=2)
    # train_poly_X = model_poly.fit_transform(X_train)
    # test_poly_X = model_poly.transform(X_test)
    #
    # model_lasso = Lasso(alpha=0.015, fit_intercept=True, max_iter=15000)
    # lasso_classifier = model_lasso.fit(train_poly_X, y_train)
    # train_pred_y = lasso_classifier.predict(train_poly_X)
    # test_pred_y = lasso_classifier.predict(test_poly_X)
    #
    # print(metrics.mean_squared_error(y_train, train_pred_y)**0.5)
    # print(metrics.mean_squared_error(y_test, test_pred_y)**0.5)



    # -------------------------------------------------------------------------------------------- #
    # Prediction of test.csv (lasso)

    # kaggle = pd.read_csv('boston_housing/test.csv')
    #
    # # Data preprocessing
    # nparr = np.array(kaggle)
    # kaggle_data = pd.DataFrame(nparr, columns=['id', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat'])
    #
    # kaggle_X = kaggle_data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
    # kaggle_X = standardizer.transform(kaggle_X)
    # kaggle_poly_X = model_poly.transform(kaggle_X)
    #
    # kaggle_pred_y = lasso_classifier.predict(kaggle_poly_X)
    #
    # print(len(kaggle_poly_X))
    # print('Final testing Prediction:', kaggle_pred_y)
    #
    # out_file(kaggle_pred_y, 'my_fourth_submission.csv')

    # ----------------------------------------------------------------------------- #
    # Fitting the Random Forest Regression to the dataset
    regressor_rf = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor_rf.fit(X_train, y_train)
    train_pred_y = regressor_rf.predict(X_train)
    test_pred_y = regressor_rf.predict(X_test)

    print('--------------------------------------------')
    print('random forest train:', metrics.mean_squared_error(y_train, train_pred_y) ** 0.5)
    print('random forest test:', metrics.mean_squared_error(y_test, test_pred_y) ** 0.5)

    print(train_pred_y)

    # -------------------------------------------------------------------------------------------- #
    # Prediction of test.csv (Random forest)

    kaggle = pd.read_csv('boston_housing/test.csv')

    # Data preprocessing
    nparr = np.array(kaggle)
    kaggle_data = pd.DataFrame(nparr, columns=['id', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat'])

    kaggle_X = kaggle_data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
    kaggle_X = standardizer.transform(kaggle_X)
    # kaggle_poly_X = model_poly.transform(kaggle_X)

    kaggle_pred_y = regressor_rf.predict(kaggle_X)

    print(len(kaggle_X))
    print('Final testing Prediction:', kaggle_pred_y)

    out_file(kaggle_pred_y, 'my_fourth_submission.csv')


def out_file(predictions, filename):
    """
    : param predictions: numpy.array, a list-like data structure that stores 0's and 1's
    : param filename: str, the filename you would like to write the results to
    """
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        start_id = 892
        for ans in predictions:
            out.write(str(start_id) + ',' + str(ans) + '\n')
            start_id += 1
    print('===============================================')


if __name__ == '__main__':
	main()
