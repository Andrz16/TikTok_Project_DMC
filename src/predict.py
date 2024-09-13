# ---------------------------------------------------------------
# Script de Entrenamiento - Modelo de 
# ---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
import pickle
import os

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = [12, 5]

# Cargar la tabla transformada
def importar_pred(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_pred = df.drop(['claim_status'],axis=1)
    y_pred = df[['claim_status']]
    return X_pred, y_pred

def importar_modelo():
    package = '../model/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    return model

def mostrar_matriz_confusion(y_val,y_pred_test):
    log_cm = confusion_matrix(y_val,y_pred_test)
    log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)
    log_disp.plot()
    plt.show()

def imprimir_metricas(y_val,y_pred_test):
    target_labels = ['opinion', 'claim']
    accuracy_test=accuracy_score(y_val,y_pred_test)
    precision_test=precision_score(y_val,y_pred_test)
    recall_test=recall_score(y_val,y_pred_test)

    print("Matriz de confusion: ")
    print("Accuracy: ", accuracy_test)
    print("Precision: ", precision_test)
    print("Recall: ", recall_test)
    print(classification_report(y_val, y_pred_test, target_names=target_labels))

def main():
    X_pred, y_pred = importar_val('data_val.csv')

    # Tokenizar las columnas
    count_vec = CountVectorizer(ngram_range=(2, 3),max_features=15,stop_words='english')
    validation_count_data = count_vec.fit_transform(X_val['video_transcription_text']).toarray()
    validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())
    validation_count_df = validation_count_df.drop(columns=['willing say'])
    X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df.drop(columns=['internet forum claim']).reset_index(drop=True)], axis=1)
    model = importar_modelo()
    y_pred_test = model.best_estimator_.predict(X_val_final)
    print('Finalizó el entrenamiento del Modelo')
    mostrar_matriz_confusion(y_val,y_pred_test)
    imprimir_metricas(y_val,y_pred_test)

if __name__ == "__main__":
    main()