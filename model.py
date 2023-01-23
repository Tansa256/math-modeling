import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
import pickle


def data_preprocessing():
    employee=pd.read_csv('Employee.csv')
    employee.rename(columns={'LeaveOrNot':'Leave'}, inplace=True)
    employee.drop_duplicates()


    categorical_features=['Education', 'City', 'PaymentTier', 'Gender', 'EverBenched']
    numerical_features=[str(feature) for feature in employee.columns if feature not in categorical_features]
    employee_encoded=pd.get_dummies(employee, columns=categorical_features, drop_first=True)
    employee_encoded.rename(columns={'EverBenched_Yes':'EverBenched', 'PaymentTier_1':'PaymentTier_Highest', 'PaymentTier_2':'PaymentTier_Middle', 'PaymentTier_3':'PaymentTier_Lowest'}, inplace=True)


    values={}
    for feature in numerical_features:
        min_val=employee_encoded[feature].min()
        max_val=employee_encoded[feature].max()
        values[feature]=[min_val, max_val]
        employee_encoded[feature]=list(map(lambda x: (x-min_val)/(max_val-min_val), employee[feature]))


    employee_df=employee_encoded.copy()


    return employee_df


def train_test(employee_df):
    X=employee_df.copy()
    X.drop('Leave', axis=1, inplace=True)
    y=employee_df['Leave']
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def model(X_train, y_train):
    lr=LogisticRegression()
    lr=lr.fit(X_train, y_train)
    return lr


df=data_preprocessing()
X_train, X_test, y_train, y_test=train_test(df)
lr=model(X_train, y_train)


y_hat_train=lr.predict(X_train)


false_positive_rate_train, true_positive_rate_train, thresholds_train = metrics.roc_curve(y_train, y_hat_train)
roc_auc_train = metrics.auc(false_positive_rate_train, true_positive_rate_train)
print('Оцінка AUC для тренувального набору: {}'.format(roc_auc_train))
print(metrics.classification_report(y_train, y_hat_train))


y_hat_test=lr.predict(X_test)


false_positive_rate_test, true_positive_rate_test, thresholds_test = metrics.roc_curve(y_test, y_hat_test)
roc_auc_test = metrics.auc(false_positive_rate_test, true_positive_rate_test)
print('Оцінка AUC для тестувального набору: {}'.format(roc_auc_test))
print(metrics.classification_report(y_test, y_hat_test))


display_train=metrics.RocCurveDisplay(fpr=false_positive_rate_train, tpr=true_positive_rate_train, roc_auc=roc_auc_train, estimator_name='ROC крива тренувального набору')
display_train.plot()


display_test=metrics.RocCurveDisplay(fpr=false_positive_rate_test, tpr=true_positive_rate_test, roc_auc=roc_auc_test, estimator_name='ROC крива тестувального набору')
display_test.plot()
plt.show()


filepath='model.sav'
pickle.dump(lr, open(filepath, 'wb'))
