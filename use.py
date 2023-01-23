import pandas as pd
import sklearn
import pickle


def load_user_file(filepath):
    df=pd.read_csv(filepath)
    return df


def save_values():
    employee=pd.read_csv('Employee.csv')
    employee.rename(columns={'LeaveOrNot':'Leave'}, inplace=True)
    employee.drop_duplicates()


    global numerical_features, categorical_features


    categorical_features=['Education', 'City', 'PaymentTier', 'Gender', 'EverBenched']
    numerical_features=[str(feature) for feature in employee.columns if (feature not in categorical_features) and (feature != 'Leave')]
    employee_encoded=pd.get_dummies(employee, columns=categorical_features, drop_first=True)
    employee_encoded.rename(columns={'EverBenched_Yes':'EverBenched', 'PaymentTier_1':'PaymentTier_Highest', 'PaymentTier_2':'PaymentTier_Middle', 'PaymentTier_3':'PaymentTier_Lowest'}, inplace=True)


    global values


    values={}
    for feature in numerical_features:
        min_val=employee_encoded[feature].min()
        max_val=employee_encoded[feature].max()
        values[feature]=[min_val, max_val]
        employee_encoded[feature]=list(map(lambda x: (x-min_val)/(max_val-min_val), employee[feature]))


    values['Education']=[[0, 0], [1, 0], [0, 1]]
    values['City']=[[0, 0], [1, 0], [0, 1]]
    values['PaymentTier']=[[0, 0], [1, 0], [0, 1]]


    return values, categorical_features, numerical_features


def normalize(value, feature, saved):
    value=(value-saved[feature][0])/(saved[feature][1]-saved[feature][0])
    return value


def create_df(file_df, saved, numerical_features, categorical_features):
    df=file_df.copy()
    del df['LeaveOrNot']
    for feature in numerical_features:
        df[feature]=list(map(lambda x: normalize(x, feature, saved), df[feature]))


    df=pd.get_dummies(df, columns=categorical_features)
    df=df.loc[:, ['JoiningYear', 'Age', 'ExperienceInCurrentDomain',  'Education_Masters', 'Education_PHD', 'City_New Delhi', 'City_Pune', 'PaymentTier_2', 'PaymentTier_3', 'Gender_Male', 'EverBenched_Yes']]
    df.rename(columns={'EverBenched_Yes':'EverBenched', 'PaymentTier_1':'PaymentTier_Highest', 'PaymentTier_2':'PaymentTier_Middle', 'PaymentTier_3':'PaymentTier_Lowest'}, inplace=True)
    return df


def predict(model, df, file_df):
    y_pred=model.predict(df)
    file_df['Prediction']=y_pred
    file_df=prediction_interpretation(file_df)
    file_df.to_csv('./user_files/prediction.csv')


def prediction_interpretation(df):
    df['Prediction']=list(map(lambda x: 'Will Leave' if int(x)==1 else 'Won`t Leave', df['Prediction']))
    return df


def user():


    model=pickle.load(open('model.sav', 'rb'))
    filepath='./user_files/to_be_predicted.csv'
    file_df=load_user_file(filepath)


    global values, categorical_features, numerical_features
    values, categorical_features, numerical_features=save_values()


    df=create_df(file_df, values, numerical_features, categorical_features)
    predict(model, df, file_df)


if __name__=='__main__':
    user()
