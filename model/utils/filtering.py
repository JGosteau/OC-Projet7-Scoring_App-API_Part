import pandas as pd


MAX_VERB_LEVEL = 3

def verbose(verb_string, verbose_level = 0):
    """
    Fonction permettant de définir plusieurs niveaux de mode verbeux.
    @retourne :
        - verb_string (Obligatoire) : (str, array of str) Indique les chaines de caractères qui seront affichées.
            + S'il s'agit d'un str : tous les niveaux du mode verbeux seront mis à cette valeur de string
            + S'il s'agit d'un array of str : chaque élement i correspond à la chaine de caractère du niveau verbeux i, si la taille 
            de l'array est inférieur à MAX_VERB_LEVEL, alors tous les autres niveau verbeux correspondront au dernier niveau définit par l'array.
        - verbose_level (Optionel) : (int) niveau du mode verbeux.
    """
    if type(verb_string) == str :
        verb_string = [verb_string] * MAX_VERB_LEVEL
    else :
        len_verb = len(verb_string)
        if len_verb < MAX_VERB_LEVEL :
            new_verb_string = ['']*MAX_VERB_LEVEL
            for i in range(len_verb) :
                new_verb_string[i] = verb_string[i]
            for i in range(len_verb, MAX_VERB_LEVEL) :
                new_verb_string[i] = verb_string[-1]
            verb_string = new_verb_string

    for i in range(MAX_VERB_LEVEL):
        if i == verbose_level :
            return verb_string[i]
    return ''

def filter_column_or_index(df,axis = 0,
                           trig_filter = 0, 
                           unit = '%',
                           verbose_level = 0,
                
                          ):
    """
    Filtre les colonnes ou lignes ayant un nombre de valeur nulle supérieur à trig_filter (en %)
    @retourne :
        - pandas.DataFrame : Dataframe filtré.
        - array : Liste des colonnes qui ont été supprimées. 
    @paramètres :
        - df (Obligatoire) : (pandas.dataframe) Echantillon étudié.
        - axis (Optionel) : (0,1,'columns', 'index') axe ou l'opération est effectuée.
        - unit (Optionel) : (str) : Si '%', alors trig_filter sera en pourcentage, sinon trig_filter représentera le nombre max de valeurs nulles
        - trig_filter (Optionel) : (float) Pourcentage / nombre de valeurs nulles toléré.
        - verbose (Optionel) : (bool) Si True, active le mode verbeux.
    """
    
    if axis == 0 or axis == 'columns' :
        index = df.index
        length = len(index)
        if unit == '%' :
            N_trig_filter = int(trig_filter/100*length) # Conversion : pourcentage du nombre de valeurs nulles --> nombre de valeurs nulles
        else :            
            N_trig_filter = trig_filter
            trig_filter = N_trig_filter/length*100 # Conversion : nombre de valeurs nulles --> pourcentage du nombre de valeurs nulles      
        count = df.count()
        count_filtered = count[count <= N_trig_filter]
        suppressed_columns = count_filtered.index
        
        # Verbose
        if unit == '%' :
            count_filtered = count_filtered/length*100 
        verb_text_level_1 = '%d colonnes sur %d, ont un nombre de valeurs nulles <= à %d/%d (%.1f%%)\n' %(len(suppressed_columns), len(df.columns), N_trig_filter, length, trig_filter)
        verb_text_level_2 = verb_text_level_1
        verb_text_level_2 += str(count_filtered.sort_values()) + '\n'
        verb = verbose(['', 
                        verb_text_level_1,
                        verb_text_level_2
                       ], verbose_level)
        print(verb, end = '')
        return df.drop(columns = suppressed_columns)
    
    if axis == 1 or axis == 'index' :
        columns = df.columns
        length = len(columns)
        if unit == '%' :
            N_trig_filter = int(trig_filter/100*length) # Conversion : pourcentage du nombre de valeurs nulles --> nombre de valeurs nulles
        else :            
            N_trig_filter = trig_filter
            trig_filter = N_trig_filter/length*100 # Conversion : nombre de valeurs nulles --> pourcentage du nombre de valeurs nulles      
            
        count = df.count(axis = 1)
        count_filtered = count[count <= N_trig_filter]
        suppressed_index = count_filtered.index
        
        # Verbose
        if unit == '%' :
            count_filtered = count_filtered/length*100 
        verb_text_level_1 = '%d lignes sur %d, ont un nombre de valeurs nulles <= à %d/%d (%.1f perc)\n' %(len(suppressed_index), len(df), N_trig_filter, length, trig_filter)
        verb_text_level_2 = verb_text_level_1
        verb_text_level_2 += str(count[count <= N_trig_filter].sort_values()) + '\n'
        verb = verbose(['',
                        verb_text_level_1,
                        verb_text_level_2,
        ], verbose_level)  
        print(verb, end = '')
        return df.drop(index = suppressed_index)