import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Bidirectional, LSTM,
        Dense, Concatenate, AveragePooling1D, MaxPooling1D,
        BatchNormalization, ReLU, Conv1D, Flatten, Lambda,
        Activation, GlobalAveragePooling1D,
        Add, Multiply, Dropout,
        Reshape, UpSampling1D)
from tensorflow.keras.callbacks import Callback

from config import kernel_size, layer_n, dropout_rate

def CMR_module(input_1, input_2, kernel_size=3, he_initializer=None, last_cmr_output=None, is_se=True):
    def _channel_attention_block(x1, x2):
        print("With channel attention")
        x = Concatenate()([x1, x2])  # (None, 2000, 128)
        x = GlobalAveragePooling1D()(x)
        x = Dense(x1.shape[-1]//8, activation="relu", kernel_initializer=he_initializer)(x)
        attention = Dense(x1.shape[-1], activation="sigmoid", kernel_initializer=he_initializer)(x)
        return attention

    def _attention_branch(x1, x2):
        print("With space attention")
        x = Concatenate()([x1, x2])
        x = Conv1D(x.shape[-1]//4, kernel_size=1, padding='same', kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(x.shape[-1], kernel_size=16, padding='same', kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(2, kernel_size=1, padding='same', kernel_initializer=he_initializer)(x)
        x = Activation('sigmoid')(x)
        return x

    attention = _attention_branch(input_1, input_2)  # Generate attention map
    attention_1 = Lambda(lambda x: x[..., 0:1])(attention)
    attention_2 = Lambda(lambda x: x[..., 1:2])(attention)
    weighted_1 = Multiply()([input_1, attention_1])
    weighted_2 = Multiply()([input_2, attention_2])

    # Calculate channel attention mechanism
    if is_se:
        channel_attention = _channel_attention_block(input_1, input_2) # (None, 64)
        weighted_1 = Multiply()([weighted_1, channel_attention]) # (None, 2000, 64)
        weighted_2 = Multiply()([weighted_2, channel_attention])
    '''
    
    x = BatchNormalization()(x)
    
       # Add residual connection
    if input_1.shape[-1] == output.shape[-1]:
        output = Add()([output, input_1])
    '''
    # Feature fusion and final processing
    if last_cmr_output is not None:
        concat = Concatenate()([weighted_1, weighted_2, last_cmr_output])
    else:
        concat = Concatenate()([weighted_1, weighted_2])

    # Final processing module (Concat-BN-ReLU-Conv)
    x = BatchNormalization()(concat)
    x = ReLU()(x)
    output = Conv1D(input_1.shape[-1], kernel_size=kernel_size, padding='same', kernel_initializer=he_initializer)(x)
    
    # output = Add()([output, last_cmr_output])
    
    return output

def feature_fusion_att(input_1, input_2, he_initializer, r=4, last_output=None, layer_n=None):
    channels = input_1.shape[-1]
    inner_channels = channels // r
    def _global_att(x):
        x_channel = x.shape[-1]
        x = GlobalAveragePooling1D()(x)
        x = Reshape((1, x_channel))(x)
        x = Conv1D(inner_channels, kernel_size=1, dilation_rate=1, strides=1, padding="valid", \
                   kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(channels, kernel_size=1, dilation_rate=1, strides=1, padding="valid", \
                   kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        return x
    def _local_att(x):
        x = Conv1D(inner_channels, kernel_size=1, dilation_rate=1, strides=1, padding="valid", \
                   kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(channels, kernel_size=1, dilation_rate=1, strides=1, padding="valid", \
                   kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        return x

    x = Add()([input_1, input_2])
    x_local = _local_att(x)
    x_global = _global_att(x)
    x = Add()([x_local, x_global])
    weight = Activation('sigmoid')(x)

    x = Add()([Multiply()([input_1, weight]),  Multiply()([input_2, (1 - weight)])])

    if last_output is not None:
        # Merge the previous output and current output to get the final output
        if last_output.shape != x.shape:
            last_output = AveragePooling1D(pool_size=5)(last_output)
            last_output = Conv1D(filters=layer_n, kernel_size=1, padding='same', kernel_initializer=he_initializer)(last_output)
        x_output = Concatenate()([x, last_output])
        x_output = BatchNormalization()(x_output)
        x_output = ReLU()(x_output)
        x_output = Conv1D(channels, kernel_size=1, padding='same', kernel_initializer=he_initializer)(x_output)
    else:
        x_output = x

    return x_output

def down_sampling(x, layer_n=64, kernel_size=7, depth=2, stride=1, he_initializer=None, x_concat=None):
    def conv_block(x, out_layer, kernel, stride, dilation):
        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same", kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def se_block(x_in, layer_n):
        x = GlobalAveragePooling1D()(x_in)
        x = Dense(layer_n//8, activation="relu", kernel_initializer=he_initializer)(x)
        x = Dense(layer_n, activation="sigmoid", kernel_initializer=he_initializer)(x)
        x_out=Multiply()([x_in, x])
        return x_out

    def res_block(x_in, layer_n, kernel, dilation, use_se=True):
        x = conv_block(x_in, layer_n, kernel, 1, dilation)
        x = conv_block(x, layer_n, kernel, 1, dilation)
        if use_se:
            x = se_block(x, layer_n)
        x = Add()([x_in, x])
        return x

    if x_concat is not None:
        x = Concatenate()([x, x_concat])
    x = conv_block(x, layer_n, kernel_size, stride, 1)
    for i in range(depth):
        x = res_block(x, layer_n, kernel_size, 1)
    return x

def cbr(x, out_layer, kernel, stride, dilation, he_initializer):
        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same", kernel_initializer=he_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

def dual_stream_model():
    he_initializer = tf.keras.initializers.HeNormal()
    # Define input
    input_shape = (2000, 1)  # Sequence length 2000, 1 dimension per step
    ecg_input = Input(shape=input_shape, name="ECG_Input")
    pcg_input = Input(shape=input_shape, name="PCG_Input")
    
    ecg_input_1 = AveragePooling1D(5)(ecg_input)
    ecg_input_2 = AveragePooling1D(25)(ecg_input)
    pcg_input_1 = AveragePooling1D(5)(pcg_input)
    pcg_input_2 = AveragePooling1D(25)(pcg_input)

#     # First BiLSTM layer
#     bilstm_1_ecg = Bidirectional(LSTM(32, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer), name="BiLSTM_Layer_1")(ecg_input)
#     bilstm_1_pcg = Bidirectional(LSTM(32, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer), name="BiLSTM_Layer_1_PCG")(pcg_input)

#     # Second BiLSTM layer (extract global features)
#     bilstm_2_ecg = Bidirectional(LSTM(64, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer), name="BiLSTM_Layer_2")(bilstm_1_ecg)
#     bilstm_2_pcg = Bidirectional(LSTM(64, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer), name="BiLSTM_Layer_2_PCG")(bilstm_1_pcg)
#     # BILSTM output is (Batch_size, 2000, 128)

#     bilstm_2_ecg_1 = AveragePooling1D(5)(bilstm_2_ecg)
#     bilstm_2_ecg_2 = AveragePooling1D(25)(bilstm_2_ecg)
#     bilstm_2_pcg_1 = AveragePooling1D(5)(bilstm_2_pcg)
#     bilstm_2_pcg_2 = AveragePooling1D(25)(bilstm_2_pcg)
    
    down_1_ecg = down_sampling(ecg_input, layer_n=layer_n*1, kernel_size=kernel_size, depth=2, stride=1, he_initializer=he_initializer)
    down_1_pcg = down_sampling(pcg_input, layer_n=layer_n*1, kernel_size=kernel_size, depth=2, stride=1, he_initializer=he_initializer)
    cmr_1 = CMR_module(down_1_ecg, down_1_pcg, kernel_size=kernel_size, he_initializer=he_initializer)
    cmr_1_x = AveragePooling1D(pool_size=5)(cmr_1)
    cmr_1_x = Conv1D(filters=layer_n*2, kernel_size=1, padding='same', kernel_initializer=he_initializer)(cmr_1_x)
    # ffa_1 = feature_fusion_att(down_1_ecg, down_1_pcg, he_initializer)

    down_2_ecg = down_sampling(down_1_ecg, layer_n=layer_n*2, kernel_size=kernel_size, depth=2, stride=5, he_initializer=he_initializer)
    down_2_pcg = down_sampling(down_1_pcg, layer_n=layer_n*2, kernel_size=kernel_size, depth=2, stride=5, he_initializer=he_initializer)
    cmr_2 = CMR_module(down_2_ecg, down_2_pcg, kernel_size=kernel_size, he_initializer=he_initializer, last_cmr_output=cmr_1_x)
    cmr_2_x = AveragePooling1D(pool_size=5)(cmr_2)
    cmr_2_x = Conv1D(filters=layer_n*3, kernel_size=1, padding='same', kernel_initializer=he_initializer)(cmr_2_x)
    # ffa_2 = feature_fusion_att(down_2_ecg, down_2_pcg, he_initializer, last_output=ffa_1, layer_n=layer_n*2)

    down_3_ecg = down_sampling(down_2_ecg, layer_n=layer_n*3, kernel_size=kernel_size, depth=2, stride=5, he_initializer=he_initializer, x_concat=ecg_input_1)
    down_3_pcg = down_sampling(down_2_pcg, layer_n=layer_n*3, kernel_size=kernel_size, depth=2, stride=5, he_initializer=he_initializer, x_concat=pcg_input_1)
    cmr_3 = CMR_module(down_3_ecg, down_3_pcg, kernel_size=kernel_size, he_initializer=he_initializer, last_cmr_output=cmr_2_x)
    cmr_3_x = AveragePooling1D(pool_size=5)(cmr_3)
    cmr_3_x = Conv1D(filters=layer_n*4, kernel_size=1, padding='same', kernel_initializer=he_initializer)(cmr_3_x)
    # ffa_3 = feature_fusion_att(down_3_ecg, down_3_pcg, he_initializer, last_output=ffa_2, layer_n=layer_n*3)

    down_4_ecg = down_sampling(down_3_ecg, layer_n=layer_n*4, kernel_size=kernel_size, depth=2, stride=5, he_initializer=he_initializer, x_concat=ecg_input_2)
    down_4_pcg = down_sampling(down_3_pcg, layer_n=layer_n*4, kernel_size=kernel_size, depth=2, stride=5, he_initializer=he_initializer, x_concat=pcg_input_2)
    cmr_4 = CMR_module(down_4_ecg, down_4_pcg, kernel_size=kernel_size, he_initializer=he_initializer, last_cmr_output=cmr_3_x)
    # cmr_4 is (None, 16, 256)
    # ffa_4 = feature_fusion_att(down_4_ecg, down_4_pcg, he_initializer, last_output=ffa_3, layer_n=layer_n*4)
    
    x = BatchNormalization()(cmr_4)
    x = ReLU()(x)

    # # First BiLSTM layer
    # bilstm_1 = Bidirectional(LSTM(256, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer), name="BiLSTM_Layer_1")(cmr_4)
    # bilstm_1 = Dropout(dropout_rate)(bilstm_1)
    # # Second BiLSTM layer (extract global features)
    # bilstm_2 = Bidirectional(LSTM(128, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer), name="BiLSTM_Layer_2")(bilstm_1)
    # bilstm_2 = Dropout(dropout_rate)(bilstm_2)

    # Dimension reduction convolution
    x = Conv1D(256, kernel_size=3, strides=2, padding='same', kernel_initializer=he_initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

#     # Temporal feature extraction
#     x = LSTM(128, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer)(x)
#     x = Dropout(dropout_rate)(x)

#     x = LSTM(64, return_sequences=True, dtype=tf.float32, kernel_initializer=he_initializer)(x)
#     x = Dropout(dropout_rate)(x)

    # # # Attention mechanism
    # # attention = Dense(1, activation='tanh', kernel_initializer=he_initializer)(x)  # (batch_size, time_steps, 1)
    # # attention = tf.keras.layers.Softmax(axis=1)(attention)  # Apply softmax along time dimension
    # # x = Multiply()([x, attention])  # Weighted features

    # Global pooling
    x = GlobalAveragePooling1D()(x)  # Replace Flatten

    # Classification head
    x = Dense(128, activation='relu', kernel_initializer=he_initializer)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu', kernel_initializer=he_initializer)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_initializer=he_initializer)(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(2, activation='softmax', kernel_initializer=he_initializer)(x)

    model = Model(inputs=[ecg_input, pcg_input], outputs=output)
    # model.summary()
    return model