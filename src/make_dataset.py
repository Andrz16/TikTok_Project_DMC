# ---------------------------------------------------------------
# Script de Preparación de Datos
# ---------------------------------------------------------------
import pandas as pd
import os
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def preparar_datos(data):
    data = data.dropna(axis=0)
    data['text_length'] = data['video_transcription_text'].str.len()
    data = data.drop(['#', 'video_id'], axis=1)
    data['claim_status'] = data['claim_status'].replace({'opinion': 0, 'claim': 1})
    data = pd.get_dummies(data,columns=['verified_status', 'author_ban_status'],drop_first=True)
    return data

def dividir_datos(data):
    data_list = []
    # Dividir los datos en conjuntos de entrenamiento y de prueba
    data_tr, data_test = train_test_split(data, test_size=0.2, random_state=0)

    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación
    data_train, data_val = train_test_split(data_tr, test_size=0.25, random_state=0)

    data_list.append(data_train)
    data_list.append(data_test)
    data_list.append(data_val)
    return data_list

def exportar_datos(data_list):
    data_list[0].to_csv(os.path.join('../data/processed/data_train.csv'))
    data_list[1].to_csv(os.path.join('../data/processed/data_test.csv'))
    data_list[2].to_csv(os.path.join('../data/processed/data_val.csv'))

def main():
    # Importar archivo
    data = pd.read_csv("../data/raw/tiktok_dataset.csv")
    # Preparar datos
    data = preparar_datos(data)
    # Dividir datos en entrenamiento, prueba y validación
    data_list = dividir_datos(data)
    # Exportar datos
    exportar_datos(data_list)

if __name__ == "__main__":
    main()