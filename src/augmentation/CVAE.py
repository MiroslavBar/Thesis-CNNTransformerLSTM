import logging as log

import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, History
from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from mne.decoding import Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.preprocessing.OneHotLabelEncoder import OneHotLabelEncoder
from src.utils import visualization, file_manager, process_utils


class CVAE(AugmentationMethod):

    def __init__(self):
        self._intermediate_dimension = 256
        self._latent_dimension = 2
        self._learning_rate = 1e-4
        self._epochs = 100
        self._validation_split = 0.15
        pipeline = file_manager.load_pickle_object(self.get_name(), 'scaling_pipeline')
        self._loaded_scaler = pipeline is not None
        self._scaling_pipeline = pipeline if pipeline else make_pipeline(Vectorizer(), MinMaxScaler())
        self._encoder_name = 'encoder'
        self._encoder_fold_id = 0
        self._decoder_name = 'decoder'
        self._decoder_fold_id = 1
        self._cvae_fold_id = 2
        self._loaded_pretrained_models = False
        self._one_hot_encoder = OneHotLabelEncoder()
        self._custom_objects = {'_sampling': self._sampling}

    def generate(self, data: np.ndarray, labels: np.ndarray, amount_to_generate: int) \
            -> tuple[np.ndarray, np.ndarray]:
        log.info(f"Using {self.get_name()} model to generate {amount_to_generate} new data points.")
        data = self._preprocess_dataset(data, labels)

        process_utils.run_in_separate_process(self._train, data)
        models = self._load_trained_models()
        history = file_manager.load_temporary_fold_history(self._cvae_fold_id)

        self._save_models(models)

        data_generated = self._generate_new_data(amount_to_generate, np.unique(labels))

        self._save_plots(models, history, (data, data_generated))

        self._remove_temporary_models()

        return data_generated

    def _preprocess_dataset(self, data: np.ndarray, labels: np.ndarray) -> tuple:
        one_hot_labels = self._one_hot_encoder.fit_transform(labels)
        if self._loaded_scaler:
            data = self._scaling_pipeline.transform(data)
        else:
            data = self._scaling_pipeline.fit_transform(data)
            file_manager.save_pickle_object(self.get_name(), 'scaling_pipeline', self._scaling_pipeline)

        return data, one_hot_labels

    def _save_plots(self, models: tuple, train_history: History or dict, data: tuple) -> None:
        data_train, data_generated = data
        x_train, y_train = data_train
        y_train_labels = self._one_hot_encoder.inverse_transform(y_train)

        self._plot_models(models)
        visualization.plot_training_history(train_history, self.get_name())

        train_reconstructed = process_utils.model_predict([x_train, y_train], self._cvae_fold_id,
                                                          remove=False, custom_objects=self._custom_objects)
        visualization.plot_real_vs_generated(
            (self._scaling_pipeline.inverse_transform(x_train), y_train_labels),
            (self._scaling_pipeline.inverse_transform(train_reconstructed), y_train_labels),
            self.get_name(),
            'train_reconstruction_compare'
        )

        original_data = self._scaling_pipeline.inverse_transform(x_train)

        visualization.plot_real_vs_generated((original_data, y_train_labels), data_generated, self.get_name())

    def _generate_new_data(self, amount_to_generate: int, unique_labels: np.ndarray) -> tuple:
        amount_to_generate_per_label = int(amount_to_generate / len(unique_labels))

        generated_data = []
        generated_data_labels = []
        for label in unique_labels:
            one_hot_labels = self._one_hot_encoder.transform(np.ones(amount_to_generate_per_label) * label)
            random_z = np.random.randn(amount_to_generate_per_label, self._latent_dimension)

            x_predicted = process_utils.model_predict([random_z, one_hot_labels], self._decoder_fold_id,
                                                      remove=False)

            x_generated = self._scaling_pipeline.inverse_transform(x_predicted)
            x_labels = np.array([label for _ in range(amount_to_generate_per_label)])
            generated_data.append(x_generated)
            generated_data_labels.append(x_labels)

        return np.concatenate(generated_data), np.concatenate(generated_data_labels)

    def _get_models(self, data_train: tuple) -> tuple:
        encoder = file_manager.load_model(self.get_name(), self._encoder_name, custom_objects=self._custom_objects)
        decoder = file_manager.load_model(self.get_name(), self._decoder_name)
        cvae = file_manager.load_model(self.get_name(), custom_objects=self._custom_objects)

        if encoder is not None and decoder is not None and cvae is not None:
            self._loaded_pretrained_models = True
            return encoder, decoder, cvae

        data_dimension = data_train[0].shape[-1]
        label_dimension = data_train[1].shape[-1]

        encoder_data_input = Input(shape=data_dimension, name='encoder_data_input')
        encoder_label_input = Input(shape=label_dimension, name='encoder_label_input')
        encoder_input = Concatenate()([encoder_data_input, encoder_label_input])
        encoder_hidden = Dense(self._intermediate_dimension, activation='relu', name='encoder_hidden')(encoder_input)

        z_mean = Dense(self._latent_dimension, name='z_mean')(encoder_hidden)
        z_log_sigma = Dense(self._latent_dimension, name='z_log_sigma')(encoder_hidden)
        z = Lambda(self._sampling, name='sampling')([z_mean, z_log_sigma])

        encoder = Model([encoder_data_input, encoder_label_input], [z_mean, z_log_sigma, z], name=self._encoder_name)

        decoder_latent_input = Input(shape=self._latent_dimension, name='decoder_latent_input')
        decoder_label_input = Input(shape=label_dimension, name='decoder_label_input')
        decoder_input = Concatenate()([decoder_latent_input, decoder_label_input])
        decoder_hidden = Dense(self._intermediate_dimension, activation='relu', name='decoder_hidden')(decoder_input)
        decoder_output = Dense(data_dimension, activation='sigmoid', name='decoder_output')(decoder_hidden)
        decoder = Model([decoder_latent_input, decoder_label_input], decoder_output, name=self._decoder_name)

        vae_output = decoder([z, encoder_label_input])
        cvae = Model([encoder_data_input, encoder_label_input], vae_output, name=self.get_name())

        loss = self._vae_loss(encoder_data_input, vae_output, z_mean, z_log_sigma)

        cvae.add_loss(loss)
        cvae.compile(optimizer=Adam(learning_rate=self._learning_rate), metrics=['mse'])

        return encoder, decoder, cvae

    def _plot_models(self, models: tuple) -> None:
        for model in models:
            visualization.save_model_graph(model, self.get_name())

    @staticmethod
    def _vae_loss(data_true, data_predicted, z_mean, z_log_sigma):
        reconstruction_loss = K.sum(K.binary_crossentropy(data_true, data_predicted), axis=-1)
        kl_loss = -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        loss = reconstruction_loss + kl_loss

        return loss

    def _sampling(self, args: tuple):
        z_mean, z_log_sigma = args
        batch_size = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, self._latent_dimension))

        return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

    def _train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        data_train = (x_train, y_train)
        models = self._get_models(data_train)

        if self._loaded_pretrained_models:
            return None

        encoder, decoder, cvae = models

        log.info(f"Training {self.get_name()} data augmentation model.")
        x_train, y_train = data_train
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        history = cvae.fit([x_train, y_train],
                           x_train,
                           shuffle=True,
                           epochs=self._epochs,
                           validation_split=self._validation_split,
                           callbacks=[early_stopping])

        file_manager.save_temporary_fold_model(encoder, self._encoder_fold_id)
        file_manager.save_temporary_fold_model(decoder, self._decoder_fold_id)
        file_manager.save_temporary_fold_model(cvae, self._cvae_fold_id)
        file_manager.save_temporary_fold_history(history, self._cvae_fold_id)

    def _save_models(self, models: tuple[Model, Model, Model]) -> None:
        if self._loaded_pretrained_models:
            return

        for model in models:
            file_manager.save_model(self.get_name(), model)

    def _load_trained_models(self) -> tuple[Model, Model, Model]:
        if self._loaded_pretrained_models:
            encoder = file_manager.load_model(self.get_name(), self._encoder_name, custom_objects=self._custom_objects)
            decoder = file_manager.load_model(self.get_name(), self._decoder_name)
            cvae = file_manager.load_model(self.get_name(), custom_objects=self._custom_objects)
        else:
            encoder = file_manager.load_temporary_fold_model(self._encoder_fold_id, custom_objects=self._custom_objects)
            decoder = file_manager.load_temporary_fold_model(self._decoder_fold_id)
            cvae = file_manager.load_temporary_fold_model(self._cvae_fold_id, custom_objects=self._custom_objects)

        return encoder, decoder, cvae

    def _remove_temporary_models(self) -> None:
        file_manager.remove_temporary_fold_model(self._encoder_fold_id)
        file_manager.remove_temporary_fold_model(self._decoder_fold_id)
        file_manager.remove_temporary_fold_model(self._cvae_fold_id)

    def get_name(self) -> str:
        return 'cVAE'
