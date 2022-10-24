if __name__ == '__main__' :
    from model.traintest import get_train_test, create_train_test
    from model.models import Level1
    import numpy as np
    import pandas as pd
    import time
    # Sépare et stocke les données de application_train en données d'entrainement et de test.
    t0 = time.time()
    print("Sépare et stocke les données de application_train en données d'entrainement et de test.", end='')
    create_train_test()
    xtrain, xtest, ytrain, ytest = get_train_test()
    loan_rate_list = np.linspace(0,1,101)
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
    
    # Créer le modèle de niveau 1 : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 : uniquement avec les variables de application_train.", end='')
    level1 = Level1(name='level1')
    level1.fit_features(xtrain, ytrain)
    level1.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()

    # Créer le modèle de niveau 1 avec une sélection de variable : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 avec une sélection de variable (10) : uniquement avec les variables de application_train.", end='')
    level1_rfe = Level1(name='level1_rfe')
    level1_rfe.auto_feature_selection_fit(xtrain, ytrain,10)
    level1_rfe.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1_rfe.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1_rfe.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
        
    # Créer le modèle de niveau 1 avec une sélection de variable : uniquement avec les variables de application_train
    print("Créer le modèle de niveau 1 avec une sélection de variable (14) : uniquement avec les variables de application_train.", end='')
    level1_rfe = Level1(name='level1_rfe_14')
    level1_rfe.auto_feature_selection_fit(xtrain, ytrain,14)
    level1_rfe.save_predict(xtest, ytest)
    exp_cost_func, optimized_triggers = level1_rfe.exploratory_cost_func(loan_rate_list = loan_rate_list, reimb_ratio_list = [0], save = True)
    level1_rfe.save_model()
    t=time.time()
    print(pd.to_timedelta((t-t0)*1e9))
    t0 = time.time()
    
