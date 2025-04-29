from keras.models import Model
from keras.layers import Dense, Input
from tcn import TCN

def build_model(sequence_length, n_features, *args, **kwargs):
    # Only numeric input
    numeric_input = Input(shape=(sequence_length, n_features), name='numeric_input')

    # Pass through TCN
    tcn_output = TCN(padding='causal', return_sequences=False)(numeric_input)

    # Final output layer
    output = Dense(1, activation='linear')(tcn_output)

    model = Model(inputs=numeric_input, outputs=output)
    return model
