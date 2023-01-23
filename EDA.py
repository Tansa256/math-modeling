import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


employee=pd.read_csv('Employee.csv')
employee.rename(columns={'LeaveOrNot':'Leave'}, inplace=True)
employee.drop_duplicates()
employee


employee.columns


employee.describe()
employee.info()


sns.set_theme()
sns.displot(employee, x='Education', hue='Leave', col='City')
sns.displot(employee, x='Education', hue='Leave', col='Gender')


sns.displot(employee, x='JoiningYear', hue='Leave', col='City')
sns.displot(employee, x='EverBenched', hue='Leave', col='City')


sns.displot(employee, x='JoiningYear', hue='Leave', col='Gender')
sns.displot(employee, x='EverBenched', hue='Leave', col='Gender')


sns.displot(employee, x='ExperienceInCurrentDomain', hue='Leave', col='City')
sns.displot(employee, x='EverBenched', hue='Leave', col='PaymentTier')


sns.displot(employee, x='Education', hue='Leave', col='PaymentTier')


categorical_features=['Education', 'City', 'PaymentTier', 'Gender', 'EverBenched']
numerical_features=[str(feature) for feature in employee.columns if feature not in categorical_features]
employee_encoded=pd.get_dummies(employee, columns=categorical_features[:3], drop_first=False)
employee_encoded=pd.get_dummies(employee_encoded, columns=categorical_features[3:], drop_first=True)
employee_encoded.rename(columns={'EverBenched_Yes':'EverBenched', 'PaymentTier_1':'PaymentTier_Highest', 'PaymentTier_2':'PaymentTier_Middle', 'PaymentTier_3':'PaymentTier_Lowest'}, inplace=True)
employee_encoded


sns.heatmap(employee_encoded.corr())


employee_encoded.describe()


values={}
for feature in numerical_features:
    min_val=employee_encoded[feature].min()
    max_val=employee_encoded[feature].max()
    values[feature]=[min_val, max_val]
    employee_encoded[feature]=list(map(lambda x: (x-min_val)/(max_val-min_val), employee[feature]))


employee_df=employee_encoded.copy()
employee_df.describe()


X=employee_df.copy()
X.drop('Leave', axis=1, inplace=True)
y=employee_df['Leave']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)


print('Кількість тренувальних прикладів: {}'.format(X_train.shape[0]))
print('Кількість тестувальних прикладів: {}'.format(X_test.shape[0]))


print('Форма тренувальних вхідних прикладів: {}'.format(X_test.shape))
print('Форма тестувальних вхідних прикладів: {}'.format(X_test.shape))
print('Форма вихідного тренувального стовбчика: {}'.format(y_train.shape))
print('Форма вихідного тестувального стовбчика: {}'.format(y_test.shape))


