from copyreg import pickle
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer

class MasterModel(BaseEstimator) :
    """
    Classe représentative du modèle de classification adapté aux calculs de probabilité de remboursement du jeu de données https://www.kaggle.com/competitions/home-credit-default-risk/data.
    """
    def __init__(self, name='Scoring', record_path=os.path.join(os.path.dirname(__file__), 'saved_models'),random_state=123, prepro = None, classif = None, features = None, prepro_type='sklearn') :
        """
        Fonction d'initialisation de la classe
        @params:
            - name (optionel) : (str) Nom du modèle, utilisé lors de la sauvegarde du modèle.
            - record_path (optionel) : (str) Chemin utilisé lors de la sauvegarde du modèle.
            - random_state (optionel) : (int) graine de randomisation.
            - prepro (optionel) : (sklearn.base.TransformerMixin) Algorithme de preprocessing des données.
            - classif (optionel) : (sklearn.base.BaseEstimator) Algorithme de classification utilisé.
            - features (optionel) : (array) Liste de noms des variable utilisées.
            - prepro_type (optionel) : (str : 'sklearn' / 'imblearn') Indicateur du type d'algorithme de preprocessing utilisé.
        """
        self.name = name
        self.record_path = record_path
        self.features = features
        self.random_state = random_state
        self.description = {}
        self.cost_func_ = {
            'exp_cost_func' : {},
            'optimized_triggers' : {}
        }
        self.prepro_type = prepro_type
        if prepro is None :
            self.prepro = FunctionTransformer()
        else :
            self.prepro = prepro
        if classif is None :
            self.classif = RandomForestClassifier(random_state=random_state)
        else :
            self.classif = classif
        self.feature_importances = None

    def get_params(self, deep=True) :
        """
        Ecrase la fonction get_params de BaseEstimator
        """
        return {
            'random_state' : self.random_state,
            'prepro' : self.prepro,
            'classif' : self.classif,
            'features' : self.features,
            'prepro_type' : self.prepro_type
        }

    def save_model(self, method = 'pickle') :
        """
        Fonction de sauvegarde du modèle.
        @paramètres :
            - method (optionel) : (str : 'pickle' / 'joblib') Processus de sauvegarde.
        """
        if method == 'joblib' :
            filepath = os.path.join(self.record_path, self.name+'.jlib')
            import joblib
            joblib.dump(self, filepath,  compress=3)
        elif method == 'pickle' :  
            filepath = os.path.join(self.record_path, self.name+'.pkl')
            import pickle
            with open(filepath, 'wb') as handle :
                pickle.dump(self, handle, protocol=-1)

    def set_features(self, features):
        """
        Fonction modifiant les variables utilisées
        @paramètres :
            - features (Obligatoire) : (array) Nouvelle liste de noms des variables sélectionnées.
        """
        self.features = features

    def fit(self, xtrain, ytrain) :
        """
        Redéfinition de la fonction fit de BaseEstimator adaptée à notre modèle.
        @paramètre :
            - xtrain (Obligatoire) : (pandas.DataFrame) Jeu de variables d'entrainement.
            - ytrain (Obligatoire) : (array ou pandas.Series) Target d'entrainement.
        """
        if self.features is None :
            self.features = self.set_features(list(xtrain.columns))
        if self.prepro_type == 'sklearn' :
            xprepro = self.prepro.fit_transform(xtrain[self.features])
            yprepro = ytrain
        elif self.prepro_type == 'imblearn' :
            xprepro, yprepro = self.prepro.fit_resample(xtrain[self.features], ytrain)
            from sklearn.model_selection import train_test_split 
            xprepro, _, yprepro, _ = train_test_split(xprepro, yprepro, train_size=700, random_state=self.random_state)
        self.classif.fit(xprepro,yprepro)
        try :
            fi = self.classif.feature_importances_
            self.feature_importances = pd.DataFrame({'feature' : self.features, 'feature_importances' : fi}).sort_values('feature_importances', ascending=False)
        except :
            self.feature_importances = None


    def predict(self, x) :
        """
        Redéfinition de la fonction predict de BaseEstimator adaptée à notre modèle.
        @paramètre :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
        """
        if self.prepro_type == 'sklearn' :
            xprepro = self.prepro.transform(x[self.features])
        elif self.prepro_type == 'imblearn' :
            xprepro = x[self.features]
        return self.classif.predict(xprepro)

    def predict_proba(self, x) :
        """
        Redéfinition de la fonction predict_proba de BaseEstimator adaptée à notre modèle.
        @paramètre :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
        @retourne :
            - yproba (2D array) : probabilité d'appartenance à une classe
        """
        if self.prepro_type == 'sklearn' :
            xprepro = self.prepro.transform(x[self.features])
        elif self.prepro_type == 'imblearn' :
            xprepro = x[self.features]
        return self.classif.predict_proba(xprepro)

    def score(self, x, y) :
        """
        Redéfinition de la fonction score de BaseEstimator adaptée à notre modèle.
        @paramètre :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
            - y (Obligatoire) : (pandas.Series) Target cible de x.
        @retourne :
            - roc (float) : score roc
        """
        from sklearn.metrics import roc_auc_score
        yproba = self.predict_proba(x)
        roc = roc_auc_score(y, yproba[:,1])
        return roc

    def save_predict(self, x, y) :
        """
        Fonction permettant de sauvegarder dans la classe MasterModel les résultats d'une prédiction pour une utilisation ultérieur.
        @paramètres :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
            - y (Obligatoire) : (pandas.Series) Target cible de x.
        @retourne :
            - yproba (2D array) : probabilité d'appartenance à une classe
        """
        from sklearn.metrics import roc_auc_score
        yproba = self.predict_proba(x)
        self.yproba = yproba
        self.y = np.array(y)
        self.roc = roc_auc_score(self.y, self.yproba[:,1])
        return yproba

    def predict_contrib_many(self, x) :
        """
        Récupère les contributions des variables pour un modèle basé sur un algorithme de type RandomForest.
        @paramètres :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
        @retourne :
            - prediction (array) : probabilité d'appartenance à la classe 0 (remboursement).
            - bias (array) : espérance du modèle d'appartenance à la classe 0.
            - contributions (array) : influence de chaque variable sur la prédiction.            
        """
        from treeinterpreter import treeinterpreter as ti
        if self.classifier == 'RFC' :
            xprepro = self.prepro.transform(x[self.features])
            prediction, bias, contributions = ti.predict(self.classif, xprepro)
            return prediction, bias, contributions

    def create_shap_explainer(self, x) :
        """
        Crée l'explicateur shap pour l'interprétation locale des variables.
        @paramètres :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
        """
        import shap
        if self.prepro_type == 'sklearn' :
            xprepro = self.prepro.transform(x[self.features])
        elif self.prepro_type == 'imblearn' :
            xprepro = x[self.features]
        self.explainer = shap.TreeExplainer(self.classif, xprepro, model_output='probability')

    def get_shap_values(self, x) :
        """
        Retourne les indices de shapley.
        @paramètres :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
        @retourne :
            - shap (shap) : indice de shapley du jeu de données x.
        """
        if self.prepro_type == 'sklearn' :
            xprepro = self.prepro.transform(x[self.features])
        elif self.prepro_type == 'imblearn' :
            xprepro = x[self.features]
        return self.explainer(xprepro)


    def shap_contribs(self, x) :
        """
        Retourne les contributions des variables sur une prédiction via une analyse des indices de shapley.
        @paramètres :
            - x (Obligatoire) : (pandas.DataFrame) Jeu de données.
        @retourne :
            - prediction (array) : probabilité d'appartenance à la classe 0 (remboursement).
            - bias (array) : espérance du modèle d'appartenance à la classe 0.
            - contributions (array) : influence de chaque variable sur la prédiction.    
        """
        prediction = self.predict_proba(x)[0,0]
        shap_values = self.get_shap_values(x)
        bias = np.array([1-shap_values.base_values, shap_values.base_values])
        contributions = np.array([-1*shap_values.values,shap_values.values]).transpose((1,2,0))

        return prediction, bias, contributions

    def predict_contrib(self, x) :
        """
        Retourne les contributions des variables.
        @paramètres :
            - x (Obligatoire) : (pandas.DataFrame à 1D) Jeu de données.
        @retourne :
            - contrib_df (pandas.DataFrame) : Contribution des différentes variables sur un jeu de données x.
        """
        if type(self.classif) == RandomForestClassifier : 
            prediction, bias, contributions = self.predict_contrib_many(x)
        else :
            prediction, bias, contributions = self.shap_contribs(x)
        contribs = contributions[0,:,0].tolist()
        contribs.insert(0, bias[0,0])
        contribs = np.array(contribs)
        contrib_df = pd.DataFrame(data=contribs, index=["Base"] + list(self.features), columns=["Contributions"])
        prediction = contrib_df.Contributions.sum()
        contrib_df.loc["Prediction"] = prediction
        contrib_df['values'] = x.iloc[0]
        if self.feature_importances is not None :
            df = pd.concat((contrib_df.loc[['Base']],contrib_df.loc[self.feature_importances.feature], contrib_df.loc[['Prediction']]))
            contrib_df = df

        return contrib_df

    def get_conf_mat(self, y=None, x=None, yproba=None, trigger=0.5, factor = None):
        """
        Retourne la matrice de confusion en fonction du seuil choisi.
        @paramètres :
            - y (Optionel) : (array) Target Cible, récupère les valeur stockées si None.
            - x (Optionel) : (pandas.DataFrame) Jeu de données, récupère les valeur stockées si None.
            - yproba (Optionel) : (array) Probabilité des classes déterminé, récupère les valeur stockées si None.
            - trigger (Optionel) : (float) Seuil de tolérance pour la prédiction de classe.
            - factor (Optionel) : (float) facteur de multiplication de la matrice de confusion.
        @retourne :
            - conf_mat (2D array) : matrice de confusion
        """
        from sklearn.metrics import confusion_matrix, roc_auc_score
        if yproba is None :
            if y is None or x is None :
                yproba = self.yproba
                y = self.y
            else :
                yproba = self.predict(x)

        ypred = np.where(yproba[:,0] >= trigger, 0,1)
        if factor is None :
            factor = 1/len(y)
        conf_mat = confusion_matrix(y, ypred)*factor
        return conf_mat

    def cost_func(self, y=None, x=None, yproba=None, conf_mat=None, trigger = 0.5, loan_rate=0.15, reimb_ratio=0):
        """
        Retourne la fonction coût métier.
        @paramètres :
            - y (Optionel) : (array) Target Cible, récupère les valeur stockées si None.
            - x (Optionel) : (pandas.DataFrame) Jeu de données, récupère les valeur stockées si None.
            - yproba (Optionel) : (array) Probabilité des classes déterminé, récupère les valeur stockées si None.
            - conf_mat (Optionel) : (2D array) Matrice de confusion.
            - trigger (Optionel) : (float) Seuil de tolérance pour la prédiction de classe.
            - loan_rate (Optionel) : (float) Taux d'emprunt.
            - reimb_ration (Optionel) : (float) Taux de remboursement du client.
        """
        if conf_mat is None :
            conf_mat = self.get_conf_mat(y, x, yproba, trigger, factor = None)
        TP = conf_mat[0, 0]
        FN = conf_mat[1, 0]
        return loan_rate*(TP+reimb_ratio*FN)-FN

    def exploratory_cost_func(self, y=None, x=None, trigger_list=None, n_trigger = 101, loan_rate_list = [0, 0.05, 0.10, 0.15], reimb_ratio_list = [0], save = False) :
        """
        Retourne la fonction coût métier en fonction du seuil et du taux d'emprunt.
        @paramètres :
            - y (Optionel) : (array) Target Cible, récupère les valeur stockées si None.
            - x (Optionel) : (pandas.DataFrame) Jeu de données, récupère les valeur stockées si None.
            - trigger_list (Optionel) : (array) Liste des seuils de tolérance téstés pour la prédiction de classe.
            - n_trigger (Optionel) : (int) nombre de seuils utilisé si trigger_list=None.
            - loan_rate_list (Optionel) : (array) Liste des taux d'emprunt testés.
            - reimb_ratio_list (Optionel) : (array) Liste des taux de remboursement du client testés.
            - save (Optionel) : (bool) Sauvegarde le résultat dans self.cost_func_ si True.
        """
        if trigger_list is None :
            trigger_list = np.linspace(0,1,n_trigger)
        data = []
        from sklearn.metrics import confusion_matrix
        if y is None or x is None :
            yproba = self.yproba
            y = self.y
        else :
            yproba = self.predict(x)
        n_pos, n_neg = np.unique(y, return_counts=True)[1]
        for trigger in trigger_list :
            conf_mat = self.get_conf_mat(y,yproba=yproba, trigger=trigger)
            TP = conf_mat[0,0]*len(y)/n_pos
            TN = conf_mat[1,1]*len(y)/n_neg
            for loan_rate in loan_rate_list :
                for reimb_ratio in reimb_ratio_list :
                    cost = self.cost_func(conf_mat=conf_mat, loan_rate=loan_rate, reimb_ratio=reimb_ratio)
                    data.append([trigger, loan_rate, reimb_ratio, cost, TP, TN])
        exp_cost_func = pd.DataFrame(data, columns=['trigger', 'loan_rate', 'reimb_rate', 'cost', 'TP', 'TN'])
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
        """
        Retourne le seuil optimal, i.e. maximum de la fonction coût métier en fonction du taux d'emprunt.
        @paramètres :
            - y (Optionel) : (array) Target Cible, récupère les valeur stockées si None.
            - x (Optionel) : (pandas.DataFrame) Jeu de données, récupère les valeur stockées si None.
            - loan_rate (Optionel) : (float) Taux d'emprunt.
            - reimb_ration (Optionel) : (float) Taux de remboursement du client.
            - trigger_list (Optionel) : (array) Liste des seuils de tolérance téstés pour la prédiction de classe.
            - n_trigger (Optionel) : (int) nombre de seuils utilisé si trigger_list=None.
        @retourne :
            - optimal_trigger (float) : seuil maximisant la fonction coût métier.
        """
        cost_func_list = self.exploratory_cost_func(y,x, loan_rate_list=[loan_rate], reimb_ratio_list=[reimb_ratio], trigger_list=trigger_list, n_trigger = n_trigger)
        optimal_index = np.argmax(cost_func_list['cost'])
        optimal_trigger = cost_func_list['trigger'][optimal_index]
        return optimal_trigger

    def explo_rfe(self, xtrain, ytrain, n_features_to_select, xtest=None, ytest=None, ratio_n_features=0.75 , ratio_step = 0.3, switch_coeff=2, step=1, verbose = 0) :
        """
        Analyse des éliminations recursives des variables.
        @paramètres :
            - xtrain (Obligatoire) : (pandas.DataFrame) données d'entrainement.
            - ytrain (Obligatoire) : (array) Targets d'entrainement.
            - n_features_to_select (Obligatoire) : (int) Nombre de variables minimales à sélectionner.
            - xtest (Optionel) : (pandas.DataFrame) Données de test, récupère les valeur stockées si None.
            - ytest (Optionel) : (array) Target de test, récupère les valeur stockées si None.
            - ratio_n_features (Optionel) : (float) Paramètre n_features des RFE.
            - ratio_step (Optionel) : (float) Paramètre step des RFE.
            - switch_coeff (Optionel) : (int) Facteur permettant de switcher sur une recherche des variables RFE avec un step = step.
            - step (Optionel) : (int) Paramètre step du RFE final.
        """
        from sklearn.feature_selection import RFE
        from sklearn.metrics import roc_auc_score
        import time
        nc = self.features
        prev_nc = nc
        features_list = {}
        scores = {}
        iter = 1
        while len(nc) > switch_coeff*n_features_to_select :
            t0 = time.time()
            prev_nc = nc
            selec = RFE(self.classif, n_features_to_select = ratio_n_features, step = ratio_step)
            xprepro = self.prepro.transform(xtrain[prev_nc])
            selec.fit(xprepro, ytrain)
            nc = prev_nc[selec.support_]
            features_list[len(nc)] = nc
            roc = np.nan
            if xtest is not None or ytest is not None :
                xprepro = self.prepro.transform(xtest[nc])
                yproba = selec.estimator_.predict_proba(xprepro.values)
                roc = roc_auc_score(ytest,yproba[:,1])
                scores[len(nc)] = roc
            if verbose >= 1 :
                print('#(%3d)-> len(feat_before)= %3d ; len(feat_after)= %3d ; score=%.3f ; time=%4.1f' %(iter, len(prev_nc), len(nc), roc, time.time()-t0))
            iter += 1
        if len(nc) != n_features_to_select :
            prev_nc = nc
            selec = RFE(self.classif, n_features_to_select = n_features_to_select, step = step)
            selec.fit(self.prepro.transform(xtrain[prev_nc]), ytrain)
            nc = prev_nc[selec.support_]
            features_list[len(nc)] = nc
            if xtest is not None or ytest is not None :
                yproba = selec.estimator_.predict_proba(xtest[nc].values)
                roc = roc_auc_score(ytest,yproba[:,1])
                scores[len(nc)] = roc
        if verbose >= 1 :
            print('#END-> len(feat_before)= %3d ; len(feat_after)= %3d ; score=%.3f ; time=%4.1f' %(len(prev_nc), len(nc), roc, time.time()-t0))
        return nc, features_list, scores

    def refit_features(self, xtrain, ytrain, features):
        """
        Refit le modèle avec un nouveau jeu de variables.
        @paramètres :
            - xtrain (Obligatoire) : (pandas.DataFrame) données d'entrainement.
            - ytrain (Obligatoire) : (array) Targets d'entrainement.
            - features (Obligatoire) : (array) Liste des noms des variables séléctionnées.
        """
        self.features = features
        self.fit(xtrain, ytrain)

    def fit_n_features(self, xtrain, ytrain, n_features) :
        """
        Détermine les variables les plus importantes par RFE et refit le modèle avec le jeu de variables déterminé.
        @paramètres :
            - xtrain (Obligatoire) : (pandas.DataFrame) données d'entrainement.
            - ytrain (Obligatoire) : (array) Targets d'entrainement.
            - n_features (Obligatoire) : (int) Nombre de variables à déterminer.
        """
        new_features, features_list, scores = self.explo_rfe(xtrain, ytrain, n_features_to_select=n_features)
        self.refit_features(xtrain, ytrain, new_features)

