import os
import pandas as pd
import numpy as np
import time
import gc
#train_path = os.path.join(os.path.dirname(__file__),"data", "application_train.csv")
train_path = os.path.join(os.path.dirname(__file__),"data", "train.csv")
RANDOM_STATE = 123


def feature_engineering_data():
    from model.utils.preprocessing import application_train_test, bureau_and_balance, previous_applications, installments_payments, credit_card_balance, pos_cash
    t0 = time.time()*1e9
    print('App : ', end = '')
    app = application_train_test()
    app_columns = np.array(app.columns[~(app.columns == 'TRAIN')])
    t = time.time()*1e9
    print(pd.to_timedelta(t-t0))
    np.save(os.path.join(os.path.dirname(__file__),"data", 'app_columns.npy'), app_columns)

    t0 = time.time()*1e9
    print('bb : ', end = '')
    bb = bureau_and_balance()
    df = app.join(bb, how = 'left')
    del app, bb
    gc.collect()
    t = time.time()*1e9
    print(pd.to_timedelta(t-t0))

    t0 = time.time()*1e9
    print('pa : ', end = '')
    pa = previous_applications()
    df = df.join(pa, how = 'left')
    del pa
    gc.collect()
    t = time.time()*1e9
    print(pd.to_timedelta(t-t0))

    t0 = time.time()*1e9
    print('ip : ', end = '')
    ip = installments_payments()
    df = df.join(ip, how = 'left')
    del ip
    gc.collect()
    t = time.time()*1e9
    print(pd.to_timedelta(t-t0))

    t0 = time.time()*1e9
    print('pc : ', end = '')
    pc = pos_cash()
    df = df.join(pc, how = 'left')
    del pc
    gc.collect()
    t = time.time()*1e9
    print(pd.to_timedelta(t-t0))

    t0 = time.time()*1e9
    print('ccb : ', end = '')
    ccb = credit_card_balance()
    df = df.join(ccb, how = 'left')
    del ccb
    gc.collect()
    t = time.time()*1e9
    print(pd.to_timedelta(t-t0))
    t0 = time.time()*1e9

    used_cols = df.columns[df.columns != 'TRAIN']
    df_train = df[df.TRAIN == 0][used_cols]
    df_test = df[df.TRAIN == 1][used_cols]
    df_train.to_csv(os.path.join(os.path.dirname(__file__),"data", "train.csv" ), index=True, compression='gzip')
    df_test.to_csv(os.path.join(os.path.dirname(__file__),"data", "test.csv" ), index=True, compression='gzip')

    

def create_train_test(random_state = RANDOM_STATE, save = True):   
    from sklearn.model_selection import train_test_split
    from model.utils.filtering import filter_column_or_index
    from model.utils.multi_label_encoder import MultiLabelEncoder
    import gc
    app = pd.read_csv(train_path, compression='gzip')
    print('app readed')
    app = filter_column_or_index(app, trig_filter=75)
    app = filter_column_or_index(app, trig_filter=99.9999, axis = 1)
    print('app filtered')

    X = app[app.columns[~app.columns.isin(['TARGET'])]].copy()
    qualcols = X.columns[(X.dtypes == object) | (X.columns.str.contains('^FLAG'))]
    quantcols = X.columns[(X.dtypes != object) & (~X.columns.str.contains('^FLAG'))]
    X[qualcols] = X[qualcols].astype(str)
    Y = app[['TARGET']]
    used_cols = np.array(X.columns)
    unique_qualcols = X[qualcols].apply(np.unique)

    MLE = MultiLabelEncoder()
    MLE.fit(X[qualcols])

    var_models = {}
    var_models['xtrain_model'], var_models['xtest_model'], var_models['ytrain_model'], var_models['ytest_model'] = train_test_split(X,Y, train_size=8000, random_state=random_state)
    print('xtrain and xtest created')
    if save :
        for var in var_models :
            var_models[var].to_csv(os.path.join(os.path.dirname(__file__),"data", "%s.csv" %(var)), index=False, compression='gzip')
            
        np.save(os.path.join(os.path.dirname(__file__),"data", 'used_cols.npy'), used_cols)
        np.save(os.path.join(os.path.dirname(__file__),"data", 'qualcols.npy'), qualcols)
        np.save(os.path.join(os.path.dirname(__file__),"data", 'quantcols.npy'), quantcols)
        
        import pickle
        with open(os.path.join(os.path.dirname(__file__),"data", "unique_qualcols.pkl"), 'wb') as handle :
            pickle.dump(unique_qualcols, handle, protocol=4)
        with open(os.path.join(os.path.dirname(__file__), "data", 'MLE.pkl'), 'wb') as handle :
            pickle.dump(MLE, handle, protocol=4)
    
    del MLE
    gc.collect()
    return var_models['xtrain_model'], var_models['xtest_model'], var_models['ytrain_model'], var_models['ytest_model']

def general_analysis(x, y, save=True) :
    import copy
    qualcols = np.load(os.path.join(os.path.dirname(__file__), "data", 'qualcols.npy'), allow_pickle=True)
    quantcols = np.load(os.path.join(os.path.dirname(__file__), "data", 'quantcols.npy'), allow_pickle=True)
    list_func = {}
    for col in quantcols :
        list_func[col] = [
            'max','min','mean', 
            'median', ('q1', lambda x : np.quantile(x,0.25)), ('q3', lambda x : np.quantile(x,0.75)), 
            ('d1', lambda x : np.quantile(x,0.1)), ('d9', lambda x : np.quantile(x,0.9))
        ]
    for col in qualcols :
        list_func[col] = [('mode', pd.Series.mode)]

    xinfo = copy.deepcopy(x)
    xinfo['TARGET'] = 'all'
    info = xinfo.groupby('TARGET').agg(list_func)
    xinfo['TARGET'] = y
    info = pd.concat((info, xinfo.groupby('TARGET').agg(list_func)))
    info = info.swaplevel(0,1,1).sort_index(1)
    
    if save :
        info.reset_index().to_csv(os.path.join(os.path.dirname(__file__),"data", "info.csv" ), index=False, compression='gzip')
    del xinfo
    gc.collect()
    return info



def get_train_test() :
    vars = []
    for var in ['xtrain_model','xtest_model','ytrain_model','ytest_model'] :
        vars.append(pd.read_csv(os.path.join(os.path.dirname(__file__),"data", "%s.csv" %(var)), compression='gzip'))
    
    qualcols = np.load(os.path.join(os.path.dirname(__file__), "data", 'qualcols.npy'), allow_pickle=True)
    vars[0][qualcols] = vars[0][qualcols].astype(str)
    vars[1][qualcols] = vars[1][qualcols].astype(str)
    return vars