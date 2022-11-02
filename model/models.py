from copyreg import pickle
import os
import numpy as np
import pandas as pd



class MasterModel() :
    def __init__(self, name='Scoring', record_path=os.path.join(os.path.dirname(__file__), 'saved_models')) :
        self.name = name
        self.record_path = record_path
        self.model = None
        self.features = None
        self.description = {}
        self.cost_func_ = {
            'exp_cost_func' : {},
            'optimized_triggers' : {}
        }
        self.classifier = 'RFC'

    def save_model(self, method = 'pickle') :
        if method == 'joblib' :
            filepath = os.path.join(self.record_path, self.name+'.jlib')
            import joblib
            joblib.dump(self, filepath,  compress=3)
        elif method == 'pickle' :  
            filepath = os.path.join(self.record_path, self.name+'.pkl')
            import pickle
            with open(filepath, 'wb') as handle :
                pickle.dump(self, handle, protocol=-1)




    def fit(self, xtrain, ytrain) :
        if self.model is not None :
            self.model.fit(xtrain, ytrain.TARGET.ravel())


    def predict(self, x) :
        if self.model is not None :
            ypred = self.model.predict_proba(x)
            """if 'SK_ID_CURR' in x :
                index = x['SK_ID_CURR']
            else :
                index = range(len(x))
            ypred = pd.DataFrame(ypred, columns=self.model.classes_, index = index)"""
            return ypred

    def save_predict(self, x, y) :
        from sklearn.metrics import roc_auc_score
        yproba = self.predict(x)
        #self.yproba = yproba.to_dict()
        #self.y = y.to_dict()
        self.yproba = yproba
        self.y = np.array(y)
        self.roc = roc_auc_score(self.y, self.yproba[:,1])
        return yproba

    def predict_contrib_many(self, x) :
        from treeinterpreter import treeinterpreter as ti
        if self.model is not None :
            if self.classifier == 'RFC' :
                sampling_pos = np.where(np.isin(list(self.model.named_steps.keys()), 'sampling'))[0]
                if len(sampling_pos) > 0 :
                    ix_end_prepro = sampling_pos[0]
                else :
                    ix_end_prepro = -1
                prepro = self.model[:ix_end_prepro]
                classif = self.model[-1]

                xprepro = prepro.transform(x)
                prediction, bias, contributions = ti.predict(classif, xprepro)
                return prediction, bias, contributions
    
    def get_prepro_class(self) :
        sampling_pos = np.where(np.isin(list(self.model.named_steps.keys()), 'sampling'))[0]
        if len(sampling_pos) > 0 :
            ix_end_prepro = sampling_pos[0]
        else :
            ix_end_prepro = -1
        self.prepro = self.model[:ix_end_prepro]
        self.classif = self.model[-1]

    def create_shap_explainer(self, x) :
        import shap
        x_shap = self.prepro.transform(x)
        self.explainer_ = shap.TreeExplainer(self.classif, x_shap)

    def explainer(self, x) :
        x_shap =  self.prepro.transform(x)
        return self.explainer_(x_shap)


    def shap_contribs(self, x) :
        prediction = self.predict(x)
        shap_values = self.explainer( x)
        bias = shap_values.base_values
        contributions = shap_values.values
        return prediction, bias, contributions

    def predict_contrib(self, x) :
        if self.model is not None :
            if self.classifier == 'RFC' : 
                prediction, bias, contributions = self.predict_contrib_many(x)
                contribs = contributions[0,:,0].tolist()
                contribs.insert(0, bias[0,0])
            else :
                try :
                    prediction, bias, contributions = self.shap_contribs(x)
                    contribs = contributions[0,:,0].tolist()
                    contribs.insert(0, bias[0,0])
                except : 
                    prediction, bias, contributions = self.shap_contribs(x)
                    contribs = contributions[0].tolist()
                    contribs.insert(0, bias[0])
            contribs = np.array(contribs)
            contrib_df = pd.DataFrame(data=contribs, index=["Base"] + list(self.features), columns=["Contributions"])
            prediction = contrib_df.Contributions.sum()
            contrib_df.loc["Prediction"] = prediction
            contrib_df['values'] = x.iloc[0]
            return contrib_df

                



    def get_conf_mat(self, y=None, x=None, yproba=None, trigger=0.5, factor = None):
        from sklearn.metrics import confusion_matrix, roc_auc_score
        if yproba is None :
            if y is None or x is None :
                #yproba = pd.DataFrame(self.yproba)
                #y = pd.DataFrame(self.y)
                yproba = self.yproba
                y = self.y
            else :
                yproba = self.predict(x)

        #ypred = np.where(yproba[0] >= trigger, 0,1)
        ypred = np.where(yproba[:,0] >= trigger, 0,1)
        if factor is None :
            factor = 1/len(y)
        conf_mat = confusion_matrix(y, ypred)*factor
        roc = roc_auc_score(y, ypred)
        return conf_mat, roc

    def cost_func(self, y=None, x=None, yproba=None, conf_mat=None, trigger = 0.5, loan_rate=0.15, reimb_ratio=0):
        if conf_mat is None :
            conf_mat, roc = self.get_conf_mat(y, x, yproba, trigger, factor = None)
        TP = conf_mat[0, 0]
        FN = conf_mat[1, 0]
        return loan_rate*(TP+reimb_ratio*FN)-FN

    def exploratory_cost_func(self, y=None, x=None, trigger_list=None, n_trigger = 101, loan_rate_list = [0, 0.05, 0.10, 0.15], reimb_ratio_list = [0], save = False) :
        if trigger_list is None :
            trigger_list = np.linspace(0,1,n_trigger)
        data = []
        from sklearn.metrics import confusion_matrix
        if y is None or x is None :
            #yproba = pd.DataFrame(self.yproba)
            #y = pd.DataFrame(self.y)
            yproba = self.yproba
            y = self.y
        else :
            yproba = self.predict(x)
        n_pos, n_neg = np.unique(y, return_counts=True)[1]
        for trigger in trigger_list :
            conf_mat, roc = self.get_conf_mat(y,yproba=yproba, trigger=trigger)
            ypred = np.where(yproba[:,0] >= trigger, 0,1)
            #conf_mat_rel = confusion_matrix(y, ypred, normalize='true')
            TP = conf_mat[0,0]*len(y)/n_pos
            TN = conf_mat[1,1]*len(y)/n_neg
            #TP = conf_mat_rel[0,0]
            #TN = conf_mat_rel[1,1]
            for loan_rate in loan_rate_list :
                for reimb_ratio in reimb_ratio_list :
                    cost = self.cost_func(conf_mat=conf_mat, loan_rate=loan_rate, reimb_ratio=reimb_ratio)
                    data.append([trigger, loan_rate, reimb_ratio, cost, TP, TN, roc])
        exp_cost_func = pd.DataFrame(data, columns=['trigger', 'loan_rate', 'reimb_rate', 'cost', 'TP', 'TN', 'ROC'])
        exp_cost_func['loan_rate'] = exp_cost_func['loan_rate'].apply(np.round, decimals=2)
        gb = exp_cost_func.groupby(['loan_rate', 'reimb_rate'])
        tl = gb['trigger'].apply(list).iloc[0]
        optimized_triggers = gb['cost'].apply(np.argmax).apply(lambda x : tl[x])
        optimized_triggers = optimized_triggers.reset_index()
        optimized_triggers.columns = list(optimized_triggers.columns[:2])+['opt_trigger']
        optimized_triggers['cost'] = gb['cost'].apply(max).reset_index(drop=True)
        optimized_triggers['loan_rate'] = optimized_triggers['loan_rate'].apply(np.round, decimals=2)
        optimized_triggers['opt_trigger'] = optimized_triggers['opt_trigger'].apply(np.round, decimals=2)
        
        if save :
            self.cost_func_ = {
                'exp_cost_func' : exp_cost_func.reset_index(drop=True).to_dict(),
                'optimized_triggers' : optimized_triggers.to_dict()
            }
        return exp_cost_func, optimized_triggers

    def get_optimal_trigger(self, y=None, x=None, loan_rate=0.15, reimb_ratio=0, trigger_list=None, n_trigger = 101) :
        cost_func_list = self.exploratory_cost_func(y,x, loan_rate_list=[loan_rate], reimb_ratio_list=[reimb_ratio], trigger_list=trigger_list, n_trigger = n_trigger)
        optimal_index = np.argmax(cost_func_list['cost'])
        optimal_trigger = cost_func_list['trigger'][optimal_index]
        return optimal_trigger


class Level1(MasterModel):
    def __init__(self, name='Scoring Level 1', forbid_columns = ['SK_ID_CURR'], record_path=os.path.join(os.path.dirname(__file__), 'saved_models'), classifier = 'RFC') :
        self.name = name
        self.record_path = record_path
        self.description = {
            'Pre-Processing' : 'RobustScalar (quantitative) & LabelEncoder (qualitative)',
            'Sampling' : 'RandomUnderSampling',
            'Classifier' : 'RandomForest'
        }
        from model.traintest import RANDOM_STATE
        self.random_state = RANDOM_STATE
        self.forbid_columns = forbid_columns
        self.features = None
        self.classifier = classifier
        self.cost_func_ = {
            'exp_cost_func' : {},
            'optimized_triggers' : {}
        }

        self.qualcols = np.load(os.path.join(os.path.dirname(__file__),"data", 'qualcols.npy'), allow_pickle=True)
        self.quantcols = np.load(os.path.join(os.path.dirname(__file__),"data", 'quantcols.npy'), allow_pickle=True)


    def init_model(self, qualcols = [], quantcols = []) :
        from imblearn.pipeline import Pipeline
        #from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import RobustScaler, FunctionTransformer
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn.ensemble import RandomForestClassifier
        from model.utils.multi_label_encoder import MultiLabelEncoder
        import pickle
        
        with open(os.path.join(os.path.dirname(__file__),"data", 'MLE.pkl'), 'rb') as handle :
            self.MLE = pickle.load(handle)

        #labelencoders = [self.MLE.labelencoders[i] for i in ix_qualcols]
        labelencoders = self.MLE.labelencoders
        print('self.MLE fitted ?', self.MLE.fitted)
        #if len(labelencoders) > 0 :
        #    print('le[1/%d] : ' %(len(labelencoders)), labelencoders[0].classes_)
        ct = ColumnTransformer([
            ('float', RobustScaler(), quantcols),
            #('object', FunctionTransformer(), qualcols),
            ('object', MultiLabelEncoder(labelencoders = labelencoders), qualcols),
        ])
        if self.classifier == 'RFC' :
            steps = [ 
                ('column_transf', ct),
                #('sampling', RandomUnderSampler(sampling_strategy=1, random_state=self.random_state)),
                ('classifier', RandomForestClassifier(random_state=self.random_state)),
            ]
        elif self.classifier == 'LGB' : 
            import lightgbm as lgb
            steps = [ 
                ('column_transf', ct),
                #('sampling', RandomUnderSampler(sampling_strategy=1, random_state=self.random_state)),
                ('classifier', lgb.LGBMClassifier(
                                    n_estimators=10000,
                                    learning_rate=0.02,
                                    num_leaves=34,
                                    colsample_bytree=0.9497036,
                                    subsample=0.8715623,
                                    max_depth=8,
                                    reg_alpha=0.041545473,
                                    reg_lambda=0.0735294,
                                    min_split_gain=0.0222415,
                                    min_child_weight=39.3259775, 
                                    random_state = self.random_state)
                                   ),
            ]
        elif self.classifier == 'GBC' : 
            from sklearn.ensemble import GradientBoostingClassifier
            steps = [ 
                ('column_transf', ct),
                #('sampling', RandomUnderSampler(sampling_strategy=1, random_state=self.random_state)),
                ('classifier', GradientBoostingClassifier(random_state = self.random_state)),
            ]

        pipeline = Pipeline(steps=steps)
        self.model = pipeline

    def fit_features(self, xtrain, ytrain, features=None) : 
        import numpy as np
        if features is None :
            features = xtrain.columns[~xtrain.columns.isin(self.forbid_columns)]
        self.features = features
        #qualcols_all = np.load(os.path.join(os.path.dirname(__file__),"data", 'qualcols.npy'), allow_pickle=True)
        #quantcols_all = np.load(os.path.join(os.path.dirname(__file__),"data", 'quantcols.npy'), allow_pickle=True)
        
        qualcols = xtrain[features].columns[xtrain[features].columns.isin(self.qualcols)]
        quantcols = xtrain[features].columns[xtrain[features].columns.isin(self.quantcols)]
        self.init_model(qualcols, quantcols)
        self.fit(xtrain, ytrain)

    def auto_feature_selection_fit(self, xtrain, ytrain, nfeatures, step = 1, features = None) :
        from sklearn.feature_selection import RFE
        import numpy as np
        if features is None :
            features = xtrain.columns[~xtrain.columns.isin(self.forbid_columns)]

        self.fit_features(xtrain, ytrain, features = features)
        classif = self.model[-1]
        prepro = self.model[:-1]
        selector = RFE(classif, n_features_to_select = nfeatures, step = step)
        #selector.fit(*prepro.fit_resample(xtrain, ytrain.TARGET.ravel()))
        selector.fit(prepro.transform(xtrain), ytrain.TARGET.ravel())
        
        qualcols_all = np.load(os.path.join(os.path.dirname(__file__),"data", 'qualcols.npy'), allow_pickle=True)
        quantcols_all = np.load(os.path.join(os.path.dirname(__file__),"data", 'quantcols.npy'), allow_pickle=True)
        qualcols = xtrain.columns[xtrain.columns.isin(qualcols_all)]
        quantcols = xtrain.columns[xtrain.columns.isin(quantcols_all)]
        
        qualcols = features[np.isin(features,self.qualcols)]
        quantcols = features[np.isin(features,self.quantcols)]

        cols = np.array(list(quantcols) + list(qualcols))
        cols = cols[~np.isin(cols, self.forbid_columns)]
        features = cols[selector.support_]

        self.fit_features(xtrain, ytrain, features)
        self.description['RFE'] = nfeatures





