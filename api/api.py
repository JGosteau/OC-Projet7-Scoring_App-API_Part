from flask import Flask, request, jsonify
from flask_restful import Resource, Api

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

import pickle
dirpath = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved_models')
print(os.listdir(dirpath))

used_cols = np.load(os.path.join(os.path.dirname(__file__),"..", "model", "data", 'used_cols.npy'), allow_pickle=True)
qualcols = np.load(os.path.join(os.path.dirname(__file__),"..", "model", "data", 'qualcols.npy'), allow_pickle=True)
quantcols = np.load(os.path.join(os.path.dirname(__file__),"..", "model", "data", 'quantcols.npy'), allow_pickle=True)

with open(os.path.join(os.path.dirname(__file__), '..','model', 'data', 'MLE.pkl'),'rb') as handle :
    mle = pickle.load(handle)

models_list = {}
for file in os.listdir(dirpath) :
    print('Loading model stored in %s : ' %(file), end = '')
    filepath = os.path.join(dirpath, file)
    name = file.split('.')[0]
    with open(filepath, 'rb') as handle : 
        models_list[name] = pickle.load(handle)
    print('Done')

from model.traintest import get_train_test
print('Loading train and test set')
xtest = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'model', 'data', 'test.csv'), compression='gzip')[used_cols]

with open(os.path.join(os.path.dirname(__file__), '..', 'model', 'data', 'unique_qualcols.pkl'), 'rb') as handle :
    unique_qualcols = pickle.load(handle)

info_train = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'model', 'data', 'info.csv'), compression='gzip', header=[0,1], index_col=0)
info_train.index = info_train.index.astype(str)
server = Flask(__name__)
api = Api(server)

class GetInfo(Resource) :
    def get(self) :
        res = {} 
        for func in info_train.columns.levels[0] :
            res[func] = {}
            info_func = info_train[func]
            for i, series in info_func.iterrows() :
                res[func][i] = series.to_dict()
        return res

api.add_resource(GetInfo, '/api/datainfo')

class GetListCols(Resource) :
    def get(self) :
        return {
            'all' : list(used_cols),
            'qualcols' : list(qualcols),
            'quantcols' : list(quantcols)
        }   
api.add_resource(GetListCols, '/api/listcols')

class GetUniqueQualcols(Resource) :
    def get(self) :
        return unique_qualcols.apply(list).to_dict()    
api.add_resource(GetUniqueQualcols, '/api/uniquequalcols')

class GetMedian(Resource) :
    def get(self) :
        return {
            'median' : info_train['median'].loc['all'].to_dict()
        }
api.add_resource(GetMedian, '/api/median')

class GetMeans(Resource) :
    def get(self) :
        return {
            'means' : info_train['mean'].loc['all'].to_dict()
        }
api.add_resource(GetMeans, '/api/means')

class GetIDs(Resource) :
    def get(self) :
        ids = list(xtest.SK_ID_CURR)
        return {'ids' : ids}
api.add_resource(GetIDs, '/api/ids')



class Imputer(Resource) :
    def post(self) :
        x = pd.Series(index = used_cols, dtype=float)
        x['SK_ID_CURR'] = 0
        data = request.get_json(force=True)
        imputer = data['imputer']
        xjson = data["x"]
        ask_features = list(xjson.keys())
        for feat in ask_features :
            x[feat] = xjson[feat]
        if imputer == 'median' :
            for feat in x.index[~x.index.isin(ask_features)] :
                if feat in qualcols :
                    x[feat] = info_train['mode'].loc['all'][feat]
                elif feat in quantcols :
                    x[feat] = info_train['median'].loc['all'][feat]
        elif imputer == 'mean' :
            for feat in x.index[~x.index.isin(ask_features)] :
                if feat in qualcols :
                    x[feat] = info_train['mode'].loc['all'][feat]
                elif feat in quantcols :
                    x[feat] = info_train['mean'].loc['all'][feat]
        elif imputer == 'iterative imputer' :
            None
        return x.to_dict()
api.add_resource(Imputer, '/api/imputer')



class GetListModel(Resource) :
    def get(self) :
        return {
            'available models' : list(models_list.keys())
        }
api.add_resource(GetListModel, '/api/models')

class GetInfoModel(Resource) :
    def post(self) :
        data = request.get_json(force=True)
        model_name = data["model"]
        md = models_list[model_name]

        fi = md.classif.feature_importances_
        df = pd.DataFrame({'feature' : md.features, 'feature_importances' : fi})
        df = df.sort_values('feature_importances', ascending = False).reset_index(drop=True)

        res = {
            "description" : md.description,
            "features" : list(df.feature),
            "feature_importances" : list(df.feature_importances)
        }
        return res
api.add_resource(GetInfoModel, '/api/getinfomodel')

class TriggerOptimizer(Resource) :
    def post(self) :
        data = request.get_json(force=True)
        model_name = data["model"]
        md = models_list[model_name]
        return md.cost_func_
api.add_resource(TriggerOptimizer, '/api/trigger')



class GetInfoId(Resource) :
    def post(self) :
        data = request.get_json(force=True)
        id = int(data['SK_ID_CURR'])
        if id in list(xtest.SK_ID_CURR) :
            x = xtest.loc[xtest.SK_ID_CURR == id]
        else :
            print('Not Found')
            x = xtrain.iloc[[0]]
            x[x.columns] = np.nan
        return x.iloc[0].to_dict()
api.add_resource(GetInfoId, '/api/getinfoid')



class PredictId(Resource) :
    def post(self) :
        data = request.get_json(force=True)
        model_name = data["model"]
        id = int(data["id"])
        if id in list(xtest.SK_ID_CURR) :
            x = xtest.loc[xtest.SK_ID_CURR == id]
        else :
            return {
                'id_found' : False,
                'probability' : None
            }
        x[qualcols] = mle.transform(x[qualcols])
        contribs = models_list[model_name].predict_contrib(x)
        probability = contribs['Contributions']['Prediction']
        return {
            'id_found' : True,
            'probability' : probability,
            'contribs' : contribs.to_dict()
        }
api.add_resource(PredictId, '/api/predictid')

class Predict(Resource) :
    def post(self) :
        data = request.get_json(force=True)
        model_name = data["model"]
        md = models_list[model_name]
        x = pd.DataFrame(index = [0], columns=used_cols)

        xjson = data["x"]
        #ask_features = list(xjson.keys())
        """
        for feat in md.features :
            if feat not in ask_features :
                print('%s not asked --> median' %(feat))
            else :
                x[feat] = xjson[feat]"""
        for feat in md.features :
            if feat in qualcols :
                x[feat] = str(xjson[feat])
                x[feat] = mle.transform(x[[feat]])
            else :
                x[feat] = xjson[feat]
        
        contribs = models_list[model_name].predict_contrib(x)
        probability = contribs['Contributions']['Prediction']
        return {
            'probability' : probability,
            'contribs' : contribs.to_dict()
        }
api.add_resource(Predict, '/api/predict')

if __name__ == '__main__' :
    server.run(debug = False)