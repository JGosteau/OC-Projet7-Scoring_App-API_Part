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
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()

    # Créer le modèle de niveau 1 avec une sélection de variable : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 avec une sélection de variable (10) : uniquement avec les variables de application_train.", end='')
    level1_rfe = Level1(name='level1_rfe')
    level1_rfe.auto_feature_selection_fit(xtrain, ytrain,10, features = features)
    level1_rfe.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1_rfe.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1_rfe.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
        
    # Créer le modèle de niveau 1 avec une sélection de variable : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 avec une sélection de variable (14) : uniquement avec les variables de application_train.", end='')
    level1_rfe = Level1(name='level1_rfe_14')
    level1_rfe.auto_feature_selection_fit(xtrain, ytrain,14, features = features)
    level1_rfe.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1_rfe.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1_rfe.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
    
    # Créer le modèle de niveau 2 : avec historique
    print("Créer le modèle de niveau 2 : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level2')
    level1.fit_features(xtrain, ytrain)
    level1.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
    
    # Créer le modèle de niveau 2 (10) : avec historique
    print("Créer le modèle de niveau 2 (10 var) : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level2_rfe')
    level1.auto_feature_selection_fit(xtrain, ytrain,10)
    level1.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
    
    # Créer le modèle de niveau 2 (14) : avec historique
    print("Créer le modèle de niveau 2 (14 var) : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level2_rfe_14')
    level1.auto_feature_selection_fit(xtrain, ytrain,14)
    level1.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
