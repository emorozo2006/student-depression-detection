import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


#data preparation
df=pd.read_csv('Student_Depression_Dataset.csv')

df.columns=df.columns.str.lower()

str_columns=list(df.dtypes[df.dtypes=='str'].index)
for c in str_columns:
    df[c]=df[c].str.lower().str.replace(' ', '_')
df.depression=(df.depression=='yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.depression.values
y_val = df_val.depression.values
y_test = df_test.depression.values

del df_train['depression']
del df_val['depression']
del df_test['depression']

#applying np.log1p() to cost_without_study to reduce right-skewness
df['cost_without_study_log'] = np.log1p(df['cost_without_study'])
df_full_train['cost_without_study_log']=np.log1p(df_full_train['cost_without_study'])
df_train['cost_without_study_log'] = np.log1p(df_train['cost_without_study'])
df_val['cost_without_study_log'] = np.log1p(df_val['cost_without_study'])
df_test['cost_without_study_log'] = np.log1p(df_test['cost_without_study'])

#applying StandartScaler() to the numerical features
numerical=['age', 'cgpa', 'cost_without_study_log']
scaler = StandardScaler()

df_train[numerical] = scaler.fit_transform(df_train[numerical])
df_val[numerical] = scaler.transform(df_val[numerical])
df_test[numerical] = scaler.transform(df_test[numerical])

categorical=[
       'gender', 
       'family_background', 
       'relationship_status',
       'marital_status',
       'living_style', 
       'current_semester',
       'friend_circle', 
       'home_division'
       ]

dicts=df[categorical+numerical].to_dict(orient='records')

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
test_dicts = df_test[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

#training the model
rf_param_grid={
    'max_depth':[5,10,15,None],
    'n_estimators':[50,100,200],
    'min_samples_leaf':[1,2,5]
}

rf_model = RandomForestClassifier(random_state=42)

rf_grid_search = GridSearchCV(
    estimator=rf_model, 
    param_grid=rf_param_grid, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1
)

print('Training the model...')
rf_grid_search.fit(X_train, y_train)
print('Training done.')

ouput_file='rf_model.bin'
with open(ouput_file, 'wb') as f_out:
    pickle.dump((dv, scaler, rf_grid_search.best_estimator_), f_out)