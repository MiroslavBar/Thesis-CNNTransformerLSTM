import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, MaxPool2D, SpatialDropout2D, BatchNormalization, LSTM
from tensorflow.keras.layers import Flatten, InputSpec, Layer, Concatenate, AveragePooling2D, MaxPooling2D, Reshape, MaxPool2D
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, LayerNormalization, SeparableConv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.layers import TimeDistributed, Lambda, AveragePooling1D, Add, Conv1D, Multiply, DepthwiseConv1D, MultiHeadAttention
from tensorflow.keras.constraints import max_norm, unit_norm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from keras import Model, Sequential
from keras.constraints import max_norm
from keras.layers import Input,Conv2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense, DepthwiseConv2D, \
    Activation, SeparableConv2D, Conv1D, DepthwiseConv1D, SeparableConv1D, AveragePooling1D

from src.classification.keras.KerasClassifier import KerasClassifier
from src.config.Config import config
from src.preprocessing.DataRepresentation import DataRepresentation
from src.preprocessing.NDScaler import NDScaler
from src.utils import file_manager

R
class CNNTransformerLSTM(KerasClassifier):

    def _create_model(self, input_shape: tuple, num_classes: int) -> Model:
        height = input_shape[0]
        conv_depth = 2
        x_input = Input(shape=input_shape)

        if config.data_representation == DataRepresentation.TIME_FREQUENCY:

            # CNN blocks
            x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x_input)
            x = BatchNormalization()(x)
            x = DepthwiseConv2D((height, 1), depth_multiplier=conv_depth, depthwise_constraint=max_norm(1.))(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((1, 4), padding='same')(x)

            x = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((1, 4), padding='same')(x)

            x = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((1, 4), padding='same')(x)

            cnn_output = Dropout(0.5)(x)

        else:
            print('just TIME_FREQUENCY')

        # Define the positional encoding function
        def positional_encoding(seq_length, d_model):
            position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
            div_term = tf.pow(10000.0, 2.0 * tf.range(d_model // 2, dtype=tf.float32) / d_model)
            angle = tf.matmul(position, div_term[tf.newaxis, :])
            sin = tf.sin(angle)
            cos = tf.cos(angle)
            pos_encoding = tf.concat([sin, cos], axis=-1)
            return pos_encoding

        # Transformer Encoder Block
        def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate):
        # Layer Normalization
            normalized_input = LayerNormalization()(inputs)

            seq_length = tf.shape(inputs)[1]
            pos_enc = positional_encoding(seq_length, d_model=128)
            transformer_input_with_pos = normalized_input + pos_enc

            # Multi-Head Attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim
            )(transformer_input_with_pos, transformer_input_with_pos)

            # Add & Normalize
            attention_output = Add()([transformer_input_with_pos, attention_output])
            normalized_output = LayerNormalization()(attention_output)

            # Feed Forward Network
            ff_output = Dense(128, activation='relu')(normalized_output)
            ff_output = Dense(128)(ff_output)

            # Add & Normalize
            encoder_output = Add()([normalized_output, ff_output])
            normalized_encoder_output = LayerNormalization()(encoder_output)

            # Dropout
            dropout_output = Dropout(dropout_rate)(normalized_encoder_output)

            return dropout_output

        cnn_output = Reshape((-1, cnn_output.shape[-1]))(cnn_output)

        # Transformer Encoder Block
        transformer_output = transformer_encoder_block(cnn_output, num_heads=2, key_dim=32, dropout_rate=0.5)

        # LSTM block
        lstm_output = LSTM(units=128, dropout=0.5, activation='tanh', return_sequences=True)(transformer_output)

        x = Flatten()(lstm_output)
        x = Dense(128, activation='relu')(x)

        # Output layer
        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=x_input, outputs=output, name=self.get_name())

        #model.summary()
        plot_model(model, "model.png", show_shapes=True)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def get_name(self) -> str:
        return 'CNNTransformerLSTM'

    def _preprocess_dataset(self, data: tuple, augmentation_method_name: str) -> tuple:
        (x_train, y_train), (x_test, y_test) = super()._preprocess_dataset(data, augmentation_method_name)

        object_type = 'scaler'
        scaler = file_manager.load_pickle_object(self.get_name(), object_type, augmentation_method_name)
        if scaler:
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        else:
            scaler = NDScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            file_manager.save_pickle_object(self.get_name(), object_type, scaler, augmentation_method_name)

        if config.data_representation == DataRepresentation.TIME_FREQUENCY:
            # Transform the input from n_epochs, n_channels, n_frequency/(height), n_times (width)
            #                     to   n_epochs, n_frequency(height), n_times (width), n_channels
            # in order to match the input shapes of the Conv2D
            x_train = np.transpose(x_train, (0, 2, 3, 1))
            x_test = np.transpose(x_test, (0, 2, 3, 1))
        else:
            x_train = np.transpose(x_train, (0, 2, 1))
            x_test = np.transpose(x_test, (0, 2, 1))

        return (x_train, y_train), (x_test, y_test)
