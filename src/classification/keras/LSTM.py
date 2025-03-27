from keras import Model, Sequential
from keras.layers import TimeDistributed, Dense, LSTM

from src.classification.keras.KerasClassifier import KerasClassifier
from src.config.Config import config
from src.preprocessing.DataRepresentation import DataRepresentation
from src.preprocessing.NDScaler import NDScaler
from src.utils import file_manager


class LSTMClassifier(KerasClassifier):

    def _create_model(self, input_shape: tuple, num_classes: int) -> Model:
        dropout = 0.2

        layers = [
            LSTM(256, return_sequences=True, dropout=dropout),
            LSTM(256, return_sequences=True, dropout=dropout),
            LSTM(256, return_sequences=False, dropout=dropout),
            Dense(num_classes, activation='softmax')
        ]

        if config.data_representation == DataRepresentation.TIME_FREQUENCY or \
                config.data_representation == DataRepresentation.OTHER:
            layers.insert(0, TimeDistributed(LSTM(256, dropout=dropout)))

        model = Sequential(layers, name=self.get_name())

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def get_name(self) -> str:
        return 'LSTM'

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

        return (x_train, y_train), (x_test, y_test)
