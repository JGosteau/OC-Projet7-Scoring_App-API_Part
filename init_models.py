import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model.traintest import get_train_test
from model.models import MasterModel
from model.utils.filtering import filter_column_or_index
from model.utils.multi_label_encoder import MultiLabelEncoder
from IPython.display import Image
import os, sys, time


train_path = os.path.join(os.path.dirname(__file__),'model', "data", "train.csv")
RANDOM_STATE = 123

# Feature engineering sur les variables des historiques
try : 
    app = pd.read_csv(train_path, compression='gzip')
except :
    from model.traintest import feature_engineering_data
    feature_engineering_data()
    app = pd.read_csv(train_path, compression='gzip')
print('app readed')
print('forme app : ', app.shape)
app = filter_column_or_index(app, trig_filter=75)
app = filter_column_or_index(app, trig_filter=99.9999, axis = 1)
print('forme app après filtrage: ', app.shape)
app.TARGET.value_counts()

# Vectorize via LabelEncoder les variables qualitatives
X = app[app.columns[~app.columns.isin(['SK_ID_CURR','TARGET'])]]
Y = app.TARGET
qualcols = X.columns[(X.dtypes == object) | (X.columns.str.contains('^FLAG'))]
quantcols = X.columns[(X.dtypes != object) & (~X.columns.str.contains('^FLAG'))]

# Label Encoder sur les variables qualitatives
mle = MultiLabelEncoder()
mle.fitted= False
mle.fit(X[qualcols].astype(str))
X[qualcols] = mle.transform(X[qualcols].astype(str))


# Permet de faire la séparation données entrainement / données test avec un resampling RandomUnderSampler.
try :
    xtrain, xtest, ytrain, ytest = get_train_test()
except :
    from model.traintest import create_train_test
    xtrain, xtest, ytrain, ytest = create_train_test(train_size=4000, random_state=RANDOM_STATE)
app_features = np.load('model/data/app_columns.npy', allow_pickle=True)
qualcols = np.load('model/data/qualcols.npy', allow_pickle=True)
quantcols = np.load('model/data/quantcols.npy', allow_pickle=True)
xtrain[qualcols] = mle.transform(xtrain[qualcols].astype(str))
xtest[qualcols] = mle.transform(xtest[qualcols].astype(str))

from sklearn.ensemble import GradientBoostingClassifier

# Entrainement modèle basé sur GradientBoosting
features = xtrain.columns[1:]
gbc_w_source = MasterModel(name='GBC_w_source', features=features, random_state=RANDOM_STATE, classif = GradientBoostingClassifier(random_state = RANDOM_STATE))
gbc_w_source.fit(xtrain, ytrain.TARGET.ravel())
print('Score gbc_wo_source %.3f' %(gbc_w_source.score(xtest, ytest)))
# Calcul roc et feature_importances vs nbr de variables séléctionnées
new_features, features_list, scores = gbc_w_source.explo_rfe(xtrain, ytrain.TARGET.ravel(), n_features_to_select=1, xtest=xtest, ytest=ytest, ratio_n_features=0.9 , ratio_step = 0.2, switch_coeff=1, verbose = 1)


# Entrainement modèle basé sur GradientBoosting sans les scores de solvabilité externes
features_wo_source = xtrain.columns[1:][~xtrain.columns[1:].isin(['EXT_SOURCE_2','EXT_SOURCE_3'])]
gbc_wo_source = MasterModel(name='GBC_wo_source', classif=GradientBoostingClassifier(random_state=RANDOM_STATE), features=features_wo_source)
gbc_wo_source.fit(xtrain, ytrain.TARGET.ravel())
print('Score gbc_wo_source %.3f' %(gbc_wo_source.score(xtest, ytest)))
# Calcul roc et feature_importances vs nbr de variables séléctionnées pour un modèle sans les variables EXT_SOURCE_2 et 3 (scores de solvabilité)
new_features_wo_source, features_list_wo_source, scores_wo_source = gbc_wo_source.explo_rfe(xtrain, ytrain.TARGET.ravel(), n_features_to_select=1, xtest=xtest, ytest=ytest, ratio_n_features=0.9 , ratio_step = 0.2, switch_coeff=1, verbose = 1)



# Entraine modèle avec/sans scores de solvabilité avec uniquement 16 variables.
n_features = 16
feat_w_source = features_list[n_features]
feat_wo_source = features_list_wo_source[n_features]
gbc_w_source_rfe = MasterModel(name='GBC_w_source_rfe_%d'%(n_features), classif=GradientBoostingClassifier(random_state=RANDOM_STATE), features=feat_w_source)
gbc_wo_source_rfe = MasterModel(name='GBC_wo_source_rfe_%d'%(n_features), classif=GradientBoostingClassifier(random_state=RANDOM_STATE), features=feat_wo_source)

gbc_w_source_rfe.fit(xtrain, ytrain.TARGET.ravel())
gbc_wo_source_rfe.fit(xtrain, ytrain.TARGET.ravel())

score_w_source = gbc_w_source_rfe.score(xtest, ytest)
score_wo_source = gbc_wo_source_rfe.score(xtest, ytest)

print('Score du modèle à %d variables (avec score de solvabilité) : %.3f' %(n_features, score_w_source))
print('Score du modèle à %d variables (sans score de solvabilité) : %.3f' %(n_features, score_wo_source))

# Sauvergarde des données
loan_rate_list = np.linspace(0,1,101)
#for i, m in enumerate([gbc_w_source, gbc_w_source_rfe, gbc_wo_source, gbc_wo_source_rfe]):
for i, m in enumerate([gbc_w_source_rfe, gbc_wo_source_rfe]):
    print(i)
    m.save_predict(xtest, ytest)
    m.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = m.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    if m == gbc_w_source or m == gbc_w_source_rfe :
        score_solv='Oui'
    else :
        score_solv='Non'
    m.description = {
        'Classifieur' : 'GradientBoosting',
        'Nbr. Variables' : len(m.features),
        'Score Solvabilité pris en compte' : score_solv,
        #'Score ROC' : m.roc
    }
    m.save_model()

import pickle
with open(os.path.join(os.path.dirname(__file__),'model', 'data','MLE.pkl'),'wb') as handle :
    pickle.dump(mle, handle)