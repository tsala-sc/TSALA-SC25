import numpy as np

def create_sequences(X, X_progName, y, sequence_length):
    X_seq, progName_seq, y_seq = [], [], []

    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:(i + sequence_length)])
        progName_seq.append(X_progName[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length - 1])
        
    return np.array(X_seq), np.array(progName_seq), np.array(y_seq)