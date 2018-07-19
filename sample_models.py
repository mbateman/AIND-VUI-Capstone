from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Model 1: RNN + TimeDistributed Dense
    # Adding batch normalization and a time distributed layer improves the loss by ~6x!
    # Try different units here, SimpleRNN, LSTM, and GRU to see how their performance differs.
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Model 2: CNN + RNN + TimeDistributed Dense
    # These models are very powerful but have a tendency to severely overfit the data.
    # You can add dropout to combat this.
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # add dropout layer
    # dropout = Dropout(0.3)(bn_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Model 3: Deeper RNN + TimeDistributed Dense
    # Adding additional layers allows your network to capture more complex sequence representations,
    # but also makes it more prone to overfitting. You can add dropout to combat this.
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layers, each with batch normalization
    rnn_input = input_data
    dropout = Dropout(0.3)(rnn_input)
    for num in range(recur_layers):
        rnn_name = 'rnn_{}'.format(num)
        simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name=rnn_name)(dropout)
        batch_norm = BatchNormalization()(simp_rnn)
        dropout1 = Dropout(0.3)(batch_norm)
        # we set rnn_input to new output, thus allowing to chain rnn
        rnn_input = dropout1
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_input)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Model 4: Bidirectional RNN + TimeDistributed Dense
    # These models tend to converge quickly.
    # They take advantage of future information through the forward and backward processing of data.
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2)
    bidir_rnn = Bidirectional(simp_rnn)(input_data)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim)) (bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    dropout = Dropout(0.3)(bn_cnn)
    rnn_input = dropout
    for num in range(recur_layers):
        rnn_name = 'rnn_{}'.format(num)
        simp_rnn = Bidirectional(
            GRU(units, activation='relu', return_sequences=True, implementation=2, name=rnn_name))(rnn_input)
        batch_norm = BatchNormalization()(simp_rnn)
        dropout1 = Dropout(0.3)(batch_norm)
        rnn_input = dropout1
    time_dense = TimeDistributed(Dense(output_dim))(rnn_input)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model