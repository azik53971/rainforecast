def initzializer():
    
    import io
    import random
    import sys
    import zipfile
    from sklearn.preprocessing import MinMaxScaler
    import os
    import torch
    from time import time
    import pandas as pd
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    from time import sleep
    import multiprocessing
    from sequential.seq2pat import Seq2Pat

    from sklearn.metrics import classification_report
    import torch.optim as optim
    
    from torch.cuda.amp import GradScaler, autocast
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader,TensorDataset,Dataset 
    
    import warnings
    warnings.filterwarnings("ignore", category=Warning)

    
    seed=42
    #Para asegurar reproducibilidad
    random.seed(seed)  # Para Python
    np.random.seed(seed)  # Para NumPy
    torch.manual_seed(seed)  # Para PyTorch
    torch.cuda.manual_seed_all(seed)  # Para CUDA
    
    df_train = pd.read_csv('train.csv')

    features = ['temparature', 'dewpoint', 
                'humidity', 'cloud', 'windspeed', 'rainfall']
    
    df_train = df_train[features]
    
    return df_train, torch, train_test_split, Dataset, DataLoader, nn, optim, F, classification_report, MinMaxScaler, Seq2Pat

def seq2pat(df_train, train_test_split, Seq2Pat):

    from sequential.seq2pat import Seq2Pat, Attribute
    from sequential.pat2feat import Pat2Feat
    
    def create_sequences(df):
        # Calcular las diferencias entre las transacciones
        sequences = []
        for index, row in df.iterrows():
            # Calcular las diferencias entre columnas consecutivas (transacciones)
            transacciones = row.dropna().values
            transacciones = transacciones[1:-1]  # Usuwa ostatnią kolumn
            sites_fila = []
        
            for i in range(len(transacciones)):
                site = transacciones[i]
                sites_fila.append(site)  # convertir a minutos

            sequences.append(sites_fila)

        return sequences

    # Tworzenie sekwencji dla dni deszczowych (rainfall == 1)
    seq_pos = create_sequences(df_train[df_train['rainfall'] == 1])
    seq_neg = create_sequences(df_train[df_train['rainfall'] == 0])

    print((seq_pos), len(seq_neg))
    
    _, seq_pos_R, = train_test_split(seq_pos, test_size=0.33, random_state=42)

    print("Seq2Pat")

    # Tworzenie obiektów
    seq2pat_neg = Seq2Pat(sequences=seq_neg)
    seq2pat_pos = Seq2Pat(sequences=seq_pos_R)
    
    print("Po Seq2Pat")
    
    return seq2pat_pos, seq2pat_neg, seq_pos_R, seq_neg
    
def DPM(seq2pat_pos, seq2pat_neg, df_train, train_test_split, seq_pos, seq_neg):
    
    from torch.nn.utils.rnn import pad_sequence
    import torch
    from sequential.seq2pat import Seq2Pat, Attribute
    from sequential.pat2feat import Pat2Feat
    from sequential.dpm import dichotomic_pattern_mining, DichotomicAggregation
    import numpy as np
    import time
    
    t = time.time()

    min_frequency_pos=0.05
    min_frequency_neg=0.05

    # Run DPM on positive and negative patterns and return a dict of pattern aggregations
    aggregation_to_patterns = dichotomic_pattern_mining(seq2pat_pos, 
                                                        seq2pat_neg,
                                                        min_frequency_pos,
                                                        min_frequency_neg)

    print(f'DPM finished! Runtime: {time.time()-t:.0f} sec')
    for aggregation, patterns in aggregation_to_patterns.items():
        print("Aggregation: ", aggregation, " with number of patterns: ", len(patterns))
    
    sequences = seq_neg + seq_pos

    dpm_patterns = aggregation_to_patterns[DichotomicAggregation.union]
    pat2feat = Pat2Feat()

    encodings = pat2feat.get_features(sequences, dpm_patterns, drop_pattern_frequency=False)

    X_train, X_test, y_train, y_test = train_test_split(encodings, 
                                                        df_train[['rainfall']].sort_values(by='rainfall',ascending=False).iloc[:len(encodings)], 
                                                        test_size=0.2, 
                                                        random_state=42)

    seq_train = [torch.tensor(seq) for seq in X_train["sequence"].tolist()]
    feat_train = torch.tensor(X_train.iloc[:, 1:].values, dtype=torch.float32)
    label_train=torch.tensor(y_train['rainfall'].tolist())

    seq_val = [torch.tensor(seq) for seq in X_test["sequence"].tolist()]
    feat_val = torch.tensor(X_test.iloc[:, 1:].values, dtype=torch.float32)
    label_val=torch.tensor(y_test['rainfall'].tolist())

    from torch.nn.utils.rnn import pad_sequence

    # # Pad sequences using pad_sequence
    seq_train = pad_sequence(seq_train, batch_first=True, padding_value=0)
    seq_val = pad_sequence(seq_val, batch_first=True, padding_value=0)
    
    print(feat_train)
    print(X_train.head())
    
def LSTM():
    class SequenceDataset(Dataset):
        def __init__(self, sequences, features, labels = None):
            self.sequences = sequences
            self.features = features
            self.labels = labels 

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            if self.labels is None:
                return self.sequences[idx], self.features[idx]
            else:
                return self.sequences[idx], self.features[idx], self.labels[idx]


if __name__ == "__main__":
    df_train, torch, train_test_split, Dataset, DataLoader, nn, optim, F, classification_report, MinMaxScaler, Seq2Pat = initzializer()
    df_train_sample = df_train.head(1000)  # Przetestuj na mniejszym zbiorze
    seq2pat_pos, seq2pat_neg, seq_pos, seq_neg = seq2pat(df_train_sample, train_test_split, Seq2Pat)
    DPM(seq2pat_pos, seq2pat_neg, df_train, train_test_split, seq_pos, seq_neg)