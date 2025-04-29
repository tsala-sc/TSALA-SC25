from keras.models import Model
from keras.layers import Dense, Input, Embedding, Concatenate, Concatenate
from tcn import TCN

def build_model(sequence_length, n_features, progName_size, embedding_dim):

    numeric_input = Input(shape=(sequence_length, n_features), name='numeric_input')
    progName_input = Input(shape=(sequence_length,), name='progName_input')
    progName_embedding = Embedding(input_dim=progName_size+1, output_dim=embedding_dim, input_length=sequence_length)(progName_input)
    
    combined_input = Concatenate(axis=-1)([progName_embedding, numeric_input])

    tcn_output = TCN(padding='causal', return_sequences=False)(combined_input)
  
    
    output = Dense(1, activation='linear')(tcn_output)

    model = Model(inputs=[progName_input, numeric_input], outputs=output)

    return model