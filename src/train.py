# ---------------------------------------------------------------
# Script de Entrenamiento
# ---------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

import warnings
warnings.filterwarnings('ignore')

# Cargar la tabla transformada
def importar_train(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(['claim_status'],axis=1)
    y_train = df[['claim_status']]
    return X_train, y_train

def entrenar_modelo(X_train,y_train,cv_params):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    rf = RandomForestClassifier(random_state=0)
    rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')

    # Tokenizar las columnas
    count_vec = CountVectorizer(ngram_range=(2, 3),max_features=15,stop_words='english')
    count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
    count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())
    X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)
    X_train_final.drop(columns=['colleague discovered', 'discovered news'], errors='ignore', inplace=True)

    rf_mod = rf_cv.fit(X_train_final, y_train)
    print('Modelo entrenado')
    return rf_mod
    
def guardar_modelo(rf_mod):
    package = '../model/best_model.pkl'
    pickle.dump(rf_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta model')

# Entrenamiento completo
def main():
    cv_params = {'max_depth': [5, 7, None],
                'max_features': [0.3, 0.6],
                'max_samples': [0.7],
                'min_samples_leaf': [1,2],
                'min_samples_split': [2,3],
                'n_estimators': [75,100,200],
                }
    X_train, y_train = importar_train('data_train.csv')
    rf_mod = entrenar_modelo(X_train,y_train,cv_params)
    guardar_modelo(rf_mod)
    print('Finalizó el entrenamiento del Modelo')