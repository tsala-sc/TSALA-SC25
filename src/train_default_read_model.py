import numpy as np
import os
import pickle
from keras.callbacks import EarlyStopping
from keras.metrics import MeanAbsolutePercentageError
from sklearn.metrics import r2_score
from fetch_data import fetch_selected_data
from temporal_extraction import process_datetime_and_trigonometric_features
from apply_scaling import apply_scaling
from create_new_database import create_sequences
from build_model import build_model


selected_columns = ['progName', 'jobID', 'startTime', 'numProc', 'numOST', 'stripeSize', 'totalFile', 'totalIOReq',
                   'totalMetaReq', 'mdsCPUMean', 'mdsOPSMean', 'seqWritePct', 'seqReadPct',
                   'consecWritePct', 'consecReadpct', 'writeBytesTotal', 'readBytesTotal',
                   'readRateTotal', 'totalReadReq', 'totalWriteReq', 'totalOpenReq','totalSeekReq', 'totalStatReq']

source_db_path = './your.db'
table_name = 'your_table'
data = fetch_selected_data(source_db_path, table_name, selected_columns)

# Add temporal features
data = process_datetime_and_trigonometric_features(data)

# remove rows with any negative values in numeric columns
numeric_columns = data.iloc[:, data.columns.get_loc('numProc'):].select_dtypes(include=['int', 'float']).columns
data = data[(data[numeric_columns] > 0).all(axis=1)]
data = data[data['readRateTotal'] != 0]

# encode 'progName' to int
data['progName_encoded'] = data['progName'].astype('category').cat.codes

# split before scaling
split_index = int(len(data) * 0.8)
train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

split_index_val = int(len(train_data) * 0.8)
valid_data = train_data.iloc[split_index_val:].copy()
train_data = train_data.iloc[:split_index_val].copy()

trigono_columns = ['startMonthDaySin', 'startMonthDayCos', 'startDaytimeSin', 'startDaytimeCos']
scaling_params = {}

for column in train_data.columns[train_data.columns.get_loc('numProc'):]:
    if column in trigono_columns:
        continue  # handled uniformly
    elif column == 'relativeStartTime':
        scaling_params[column] = {'max': train_data[column].max()}
    elif column == 'stripeSize':
        train_data['stripeSizeRank'] = train_data['stripeSize'].rank(method='average')
        train_data['stripeSize'] = (train_data['stripeSizeRank'] - 1) / (len(train_data['stripeSize']) - 1)
        scaling_params[column] = {'denom': len(train_data['stripeSize']) - 1}
        train_data.drop('stripeSizeRank', axis=1, inplace=True)
    else:
        train_data[column] = np.log(train_data[column] + 0.01)
        scaling_params[column] = {
            'min': train_data[column].min(),
            'max': train_data[column].max()
        }
        train_data[column] = (train_data[column] - scaling_params[column]['min']) / (scaling_params[column]['max'] - scaling_params[column]['min'])

# normalize trigono columns for train
for column in trigono_columns:
    train_data[column] = (train_data[column] + 1.0) / 2.0

# save scaling params
with open("scaling_params_read.pkl", "wb") as f:
    pickle.dump(scaling_params, f)

# apply to valid/test
valid_data = apply_scaling(valid_data, scaling_params, trigono_columns)
test_data = apply_scaling(test_data, scaling_params, trigono_columns)

# extract inputs
def extract_X_y(df):
    X = df.drop(['readRateTotal','progName', 'jobID', 'progName_encoded'], axis=1).values
    X_progName = df['progName_encoded'].values
    y = df['readRateTotal'].values
    return X, X_progName, y

X_train, X_progName_train, y_train = extract_X_y(train_data)
X_valid, X_progName_valid, y_valid = extract_X_y(valid_data)
X_test, X_progName_test, y_test = extract_X_y(test_data)

# directory for saving model
model_dir = './models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# include to find the best seqeunce length
sequence_lengths = [40]
n_features = X_train.shape[1]
progName_size = data['progName'].nunique()
embedding_dim = 30

best_r_squared = 0
best_model = None
best_sequence_length = None

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for sequence_length in sequence_lengths:
    X_train_seq, X_progName_train_seq, y_train_seq = create_sequences(X_train, X_progName_train, y_train, sequence_length)
    X_valid_seq, X_progName_valid_seq,  y_valid_seq = create_sequences(X_valid, X_progName_valid, y_valid, sequence_length)
    X_test_seq, X_progName_test_seq, y_test_seq = create_sequences(X_test, X_progName_test, y_test, sequence_length)

    model = build_model(sequence_length, n_features, progName_size, embedding_dim)
    model.compile(optimizer="adam", loss="mse", metrics=MeanAbsolutePercentageError())

    model.fit([X_progName_train_seq, X_train_seq], y_train_seq, 
              epochs=100,
              validation_data=([X_progName_valid_seq, X_valid_seq], y_valid_seq), 
              callbacks=[early_stopping])

    eval_result = model.evaluate([X_progName_test_seq, X_test_seq], y_test_seq)
    current_mae = eval_result[0]
    current_mape = eval_result[1]
    print(f"Sequence length {sequence_length} MAE: {current_mae} MAPE: {current_mape}")

    predicted_data = model.predict([X_progName_test_seq, X_test_seq])
    current_r_squared = r2_score(y_test_seq, predicted_data)
    print(f"rscore: {current_r_squared}")

    if current_r_squared > best_r_squared:
        best_r_squared = current_r_squared
        best_model = model
        best_sequence_length = sequence_length

best_model_path = os.path.join(model_dir, f"tsala_system_read.h5")
best_model.save(best_model_path)
print(f"Best model with sequence length {best_sequence_length} is saved to {best_model_path}.")
