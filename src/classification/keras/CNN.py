import numpy as np
from keras import Model, Sequential
from keras.constraints import max_norm
from keras.layers import Conv2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense, DepthwiseConv2D, \
    Activation, SeparableConv2D, Conv1D, DepthwiseConv1D, SeparableConv1D, AveragePooling1D

from src.classification.keras.KerasClassifier import KerasClassifier
from src.config.Config import config
from src.preprocessing.DataRepresentation import DataRepresentation
from src.preprocessing.NDScaler import NDScaler
from src.utils import file_manager


class CNN(KerasClassifier):

    def _create_model(self, input_shape: tuple, num_classes: int) -> Model:
        height = input_shape[0]
        conv_depth = 2
        filters_block1 = 8
        filters_block2 = 16
        kernel_size_block1 = 64
        kernel_size_block2 = 16
        dropout_rate = 0.5

        if config.data_representation == DataRepresentation.TIME_FREQUENCY:
            pool_size = (1, 4)
            model = Sequential([
                # Block 1
                Conv2D(filters_block1, (1, kernel_size_block1), padding='same', input_shape=input_shape),
                BatchNormalization(),
                DepthwiseConv2D((height, 1), depth_multiplier=conv_depth, depthwise_constraint=max_norm(1.)),
                BatchNormalization(),
                Activation('relu'),
                AveragePooling2D(pool_size=pool_size),
                Dropout(dropout_rate),

                # Block 2
                SeparableConv2D(filters_block2, (1, kernel_size_block2), padding='same'),
                BatchNormalization(),
                Activation('relu'),
                AveragePooling2D(pool_size=pool_size),
                Dropout(dropout_rate),

                Flatten(),

                Dense(num_classes, activation='softmax')
            ], name=self.get_name())
        else:
            pool_size = 4
            model = Sequential([
                # Block 1
                Conv1D(filters_block1, kernel_size_block1, padding='same', input_shape=input_shape),
                BatchNormalization(),
                DepthwiseConv1D(kernel_size_block1, depth_multiplier=conv_depth, depthwise_constraint=max_norm(1.)),
                BatchNormalization(),
                Activation('relu'),
                AveragePooling1D(pool_size=pool_size),
                Dropout(dropout_rate),

                # Block 2
                SeparableConv1D(filters_block2, kernel_size_block2, padding='same'),
                BatchNormalization(),
                Activation('relu'),
                AveragePooling1D(pool_size=pool_size),
                Dropout(dropout_rate),

                Flatten(),

                Dense(num_classes, activation='softmax')
            ], name=self.get_name())

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def get_name(self) -> str:
        return 'CNN'

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
