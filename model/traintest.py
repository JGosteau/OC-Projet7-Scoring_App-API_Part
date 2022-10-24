import os
import pandas as pd
import numpy as np

train_path = os.path.join(os.path.dirname(__file__),"data", "application_train.csv")
RANDOM_STATE = 123

def create_train_test(random_state = RANDOM_STATE):   
    from sklearn.model_selection import train_test_split
    from model.utils.filtering import filter_column_or_index
    from model.utils.multi_label_encoder import MultiLabelEncoder
    app = pd.read_csv(train_path, compression='gzip')
    app = filter_column_or_index(app, trig_filter=75)
    app = filter_column_or_index(app, trig_filter=99.9999, axis = 1)

    X = app[app.columns[~app.columns.isin(['TARGET'])]].copy()
    qualcols = X.columns[(X.dtypes == object) | (X.columns.str.contains('FLAG'))]
    quantcols = X.columns[(X.dtypes != object) & (~X.columns.str.contains('FLAG'))]
    X[qualcols] = X[qualcols].astype(str)

    MLE = MultiLabelEncoder()
    MLE.fit(X[qualcols])

    Y = app.TARGET
    
    var_models = {}
    var_models['xtrain_model'], var_models['xtest_model'], var_models['ytrain_model'], var_models['ytest_model'] = train_test_split(X,Y, train_size=8000, random_state=random_state)

    for var in var_models :
        var_models[var].to_csv(os.path.join(os.path.dirname(__file__),"data", "%s.csv" %(var)), index=False, compression='gzip')
    
    used_cols = np.array(X.columns)
    np.save(os.path.join(os.path.dirname(__file__),"data", 'used_cols.npy'), used_cols)
    np.save(os.path.join(os.path.dirname(__file__),"data", 'qualcols.npy'), qualcols)
    np.save(os.path.join(os.path.dirname(__file__),"data", 'quantcols.npy'), quantcols)
    xtrain = var_models['xtrain_model']
    median_train = xtrain[quantcols].median().reset_index()
    means_train = xtrain[quantcols].mean().reset_index()
    mode_train = xtrain[qualcols].mode().reset_index()
    
    median_train.to_csv(os.path.join(os.path.dirname(__file__),"data", "median.csv" ), index=False, compression='gzip')
    means_train.to_csv(os.path.join(os.path.dirname(__file__),"data", "means.csv" ), index=False, compression='gzip')
    mode_train.to_csv(os.path.join(os.path.dirname(__file__),"data", "mode_train.csv" ), index=False, compression='gzip')

    #unique_qualcols = X[qualcols].apply(np.unique)
    unique_qualcols = X[qualcols].astype(str).apply(np.unique)
    #unique_qualcols.to_json(os.path.join(os.path.dirname(__file__),"data", "unique_qualcols.csv" ), index=False, compression='gzip')
    
    
    import pickle
    with open(os.path.join(os.path.dirname(__file__),"data", "unique_qualcols.pkl"), 'wb') as handle :
        pickle.dump(unique_qualcols, handle, protocol=4)
    with open(os.path.join(os.path.dirname(__file__), "data", 'MLE.pkl'), 'wb') as handle :
        pickle.dump(MLE, handle, protocol=4)

def get_train_test() :
    vars = []
    for var in ['xtrain_model','xtest_model','ytrain_model','ytest_model'] :
        vars.append(pd.read_csv(os.path.join(os.path.dirname(__file__),"data", "%s.csv" %(var)), compression='gzip'))
    
    qualcols = np.load(os.path.join(os.path.dirname(__file__), "data", 'qualcols.npy'), allow_pickle=True)
    vars[0][qualcols] = vars[0][qualcols].astype(str)
    vars[1][qualcols] = vars[1][qualcols].astype(str)
    return vars