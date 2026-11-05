import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.pipeline import make_pipeline

def load_data():
    df=pd.read_csv('data/Student_Depression_Dataset.csv')
    df.columns=df.columns.str.lower()
    str_columns=list(df.dtypes[df.dtypes=='str'].index)

    for c in str_columns:
        df[c]=df[c].str.lower().str.replace(' ', '_')

    df.depression=(df.depression=='yes').astype(int)
    return df


def train_model(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train['depression'].values
    y_test = df_test['depression'].values

    del df_train['depression']
    del df_test['depression']

    df_train['cost_without_study_log'] = np.log1p(df_train['cost_without_study'])
    df_test['cost_without_study_log'] = np.log1p(df_test['cost_without_study'])

    numerical = ['age', 'cgpa', 'cost_without_study_log']
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


    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    test_dicts = df_test[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    #Random Forest Model
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

    rf_grid_search.fit(X_train, y_train)

    pipeline = make_pipeline(
        DictVectorizer(sparse=False), 
        rf_grid_search.best_estimator_)

    pipeline.fit(train_dicts, y_train)

    return pipeline

def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'model saved to {filename}')


df=load_data()
pipeline=train_model(df)
save_model('models/rf_model.bin', pipeline)


