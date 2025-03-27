from keras import Model, Sequential
from keras.layers import Dense, Flatten
from mne.decoding import Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classification.keras.KerasClassifier import KerasClassifier

from src.utils import file_manager


class MLP(KerasClassifier):

    def __init__(self):
        super().__init__()
        self._epochs = 100

    def _create_model(self, input_shape: tuple, num_classes: int) -> Model:
        model = Sequential([
            Flatten(),
            Dense(512, activation='sigmoid'),
            Dense(256, activation='sigmoid'),
            Dense(128, activation='sigmoid'),
            Dense(num_classes, activation='softmax')
        ], name=self.get_name())

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def _preprocess_dataset(self, data, augmentation_method_name) -> tuple:
        (x_train, y_train), (x_test, y_test) = super()._preprocess_dataset(data, augmentation_method_name)

        object_type = 'scaling_pipeline'
        pipeline = file_manager.load_pickle_object(self.get_name(), object_type, augmentation_method_name)
        if pipeline:
            x_train = pipeline.transform(x_train)
            x_test = pipeline.transform(x_test)
        else:
            pipeline = make_pipeline(Vectorizer(), StandardScaler())
            x_train = pipeline.fit_transform(x_train)
            x_test = pipeline.transform(x_test)

            file_manager.save_pickle_object(self.get_name(), object_type, pipeline, augmentation_method_name)

        return (x_train, y_train), (x_test, y_test)

    def get_name(self) -> str:
        return 'MLP'
