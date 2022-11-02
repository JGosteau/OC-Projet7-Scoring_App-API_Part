if __name__ == '__main__' :
    from model.traintest import get_train_test, create_train_test, feature_engineering_data, general_analysis
    from model.models import Level1
    import numpy as np
    import pandas as pd
    import time
    import os

    # Feature Engineering.
    print("Feature Engineering.")
    #feature_engineering_data()
    save_files = True
    # Sépare et stocke les données de application_train en données d'entrainement et de test.
    t0 = time.time()
    print("Sépare et stocke les données de application_train en données d'entrainement et de test.")
    xtrain, xtest, ytrain, ytest = create_train_test(save = save_files)
    print('length xtrain, xtest', len(xtrain), len(xtest))
    #xtrain, xtest, ytrain, ytest = get_train_test()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))

    t0 = time.time()
    print("Récupère les moyennes /médianes, etc des données d'entrainement.")
    info = general_analysis(xtrain, ytrain, save = save_files)
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))

    loan_rate_list = np.linspace(0,1,101)
    
    app_features = np.load(os.path.join(os.path.dirname(__file__),"model", "data", 'app_columns.npy'), allow_pickle=True)
    features = app_features[np.isin(app_features, xtrain.columns)]

    
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level1')
    level1.fit_features(xtrain, ytrain, features = features)
    level1.save_predict(xtest, ytest)
    #level1.get_prepro_class()
    #level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    print('roc max : ', roc_max)
    from sklearn.metrics import roc_auc_score
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    t0 = time.time()

    # Créer le modèle de niveau 1 avec une sélection de variable : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 avec une sélection de variable (10) : uniquement avec les variables de application_train.", end='')
    level1_rfe = Level1(name='level1_rfe')
    level1_rfe.auto_feature_selection_fit(xtrain, ytrain,10, features = features)
    level1_rfe.save_predict(xtest, ytest)
    #level1_rfe.get_prepro_class()
    #level1_rfe.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1_rfe.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1_rfe.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    roc_max = max(pd.DataFrame(level1_rfe.cost_func_['exp_cost_func'])['ROC'])
    print('roc max : ', roc_max)
    from sklearn.metrics import roc_auc_score
    score_roc_1 = roc_auc_score(level1_rfe.y, level1_rfe.yproba[:,1])
    score_roc_2 = roc_auc_score(level1_rfe.y, np.where(level1_rfe.yproba[:,1]>0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1_rfe.roc))
    t0 = time.time()
        
    # Créer le modèle de niveau 1 avec une sélection de variable : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 avec une sélection de variable (14) : uniquement avec les variables de application_train.", end='')
    level1_rfe = Level1(name='level1_rfe_14')
    level1_rfe.auto_feature_selection_fit(xtrain, ytrain,14, features = features)
    level1_rfe.save_predict(xtest, ytest)
    level1_rfe.get_prepro_class()
    #level1_rfe.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1_rfe.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1_rfe.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    roc_max = max(pd.DataFrame(level1_rfe.cost_func_['exp_cost_func'])['ROC'])
    print('roc max : ', roc_max)
    from sklearn.metrics import roc_auc_score
    score_roc_1 = roc_auc_score(level1_rfe.y, level1_rfe.yproba[:,1])
    score_roc_2 = roc_auc_score(level1_rfe.y, np.where(level1_rfe.yproba[:,1]>0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1_rfe.roc))
    t0 = time.time()
    
    """
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur lightgb : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='lgb', classifier = 'LGB')
    level1.fit_features(xtrain, ytrain, features = features)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()
    
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur lightgb rfe (14): uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='lgb_rfe_14', classifier = 'LGB')
    level1.auto_feature_selection_fit(xtrain, ytrain,14, features = features)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()
    """

    
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur gbc : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='gbc', classifier = 'GBC')
    level1.fit_features(xtrain, ytrain, features = features)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()


    
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur lightgb gbc (14): uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='gbc_rfe_14', classifier = 'GBC')
    level1.auto_feature_selection_fit(xtrain, ytrain,14, features = features)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()


    # Créer le modèle de niveau 2 : avec historique
    print("Créer le modèle de niveau 2 : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level2')
    level1.fit_features(xtrain, ytrain)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    #level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    print('roc max : ', roc_max)
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    t0 = time.time()
    
    # Créer le modèle de niveau 2 (10) : avec historique
    print("Créer le modèle de niveau 2 (10 var) : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level2_rfe')
    level1.auto_feature_selection_fit(xtrain, ytrain,10)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    #level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    print('roc max : ', roc_max)
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    t0 = time.time()
    
    # Créer le modèle de niveau 2 (14) : avec historique
    print("Créer le modèle de niveau 2 (14 var) : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level2_rfe_14')
    level1.auto_feature_selection_fit(xtrain, ytrain,14)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    #level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    print('roc max : ', roc_max)
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    t0 = time.time()
    """
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur lightgb : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='lgb_hist', classifier = 'LGB')
    level1.fit_features(xtrain, ytrain)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()

    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur lightgb rfe (14): uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='lgb_rfe_14', classifier = 'LGB')
    level1.auto_feature_selection_fit(xtrain, ytrain,14)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()
    """
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur gbc rfe (14): uniquement avec hist.", end='')
    level1 = Level1(name='gbc_hist', classifier = 'GBC')
    level1.fit_features(xtrain, ytrain)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()


    
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle basé sur gbc rfe (14): uniquement avec hist.", end='')
    level1 = Level1(name='gbc_hist_rfe_14', classifier = 'GBC')
    level1.auto_feature_selection_fit(xtrain, ytrain,14)
    level1.save_predict(xtest, ytest)
    level1.get_prepro_class()
    level1.create_shap_explainer(xtrain)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    score_roc_1 = roc_auc_score(level1.y, level1.yproba[:,1])
    score_roc_2 = roc_auc_score(level1.y, np.where(level1.yproba[:,1]>=0.5,1,0))
    print('ROC_AUC 1 : %.3f\nROC_AUC 2 : %.3f, ROC : %.3f' %(score_roc_1, score_roc_2, level1.roc))
    #roc_max = max(pd.DataFrame(level1.cost_func_['exp_cost_func'])['ROC'])
    #print('roc max : ', roc_max)
    t0 = time.time()