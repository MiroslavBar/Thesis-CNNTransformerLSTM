import logging as log
from functools import partial

import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.callbacks import History
from keras.layers import Dense, LeakyReLU, \
    Flatten, Embedding, multiply, Reshape
from keras.models import Model
from keras.optimizers.optimizer_v2.adam import Adam
from keras.optimizers.optimizer_v2.rmsprop import RMSProp
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.augmentation.AugmentationMethod import AugmentationMethod
from src.preprocessing.NDScaler import NDScaler
from src.utils import file_manager, visualization, process_utils


class CWGANGP(AugmentationMethod):

    def __init__(self):
        self._generator_name = 'generator'
        self._generator_fold_id = 0
        self._critic_name = 'critic'
        self._critic_fold_id = 1
        self._noise_dimension = 100
        self._batch_size = 32
        self._critic_iterations = 5
        self._gradient_penalty_weight = 10
        self._epochs = 525
        scaler = file_manager.load_pickle_object(self.get_name(), 'scaler')
        self._loaded_scaler = scaler is not None
        self._scaler = scaler if scaler else NDScaler(MinMaxScaler(feature_range=(-1, 1)))
        self._loaded_pretrained_models = False
        self._debug_save_interval = 75
        self._debug_save = False

    def generate(self, data: np.ndarray, labels: np.ndarray, amount_to_generate) \
            -> tuple[np.ndarray, np.ndarray]:
        log.info(f"Using {self.get_name()} model to generate {amount_to_generate} new data points.")
        data = self._preprocess_dataset(data, labels)

        process_utils.run_in_separate_process(self._train, data)
        models = self._load_trained_models()
        history = file_manager.load_temporary_fold_history(self._generator_fold_id)
        self._save_models(models)

        data_generated = self._generate_new_data(amount_to_generate, np.unique(labels))

        self._save_plots(models, history, (data, data_generated))

        self._remove_temporary_models()

        return data_generated

    def _preprocess_dataset(self, data: np.ndarray, labels: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:
        if self._loaded_scaler:
            data = self._scaler.transform(data)
        else:
            data = self._scaler.fit_transform(data)
            file_manager.save_pickle_object(self.get_name(), 'scaler', self._scaler)

        return data, labels

    def _save_plots(self, models: tuple, train_history: History or dict, data: tuple):
        data_train, data_generated = data
        x_train, y_train = data_train

        self._plot_models(models)
        visualization.plot_training_history(train_history, self.get_name())

        x_train = self._scaler.inverse_transform(x_train)

        visualization.plot_real_vs_generated((x_train, y_train), data_generated, self.get_name())

    def _get_models(self, data_train: tuple[np.ndarray, np.ndarray]) -> tuple[Model, Model]:
        generator = file_manager.load_model(self.get_name(), self._generator_name)
        critic = file_manager.load_model(self.get_name(), self._critic_name)

        if generator is not None and critic is not None:
            self._loaded_pretrained_models = True
            return generator, critic

        generator = self._generator(data_train)
        critic = self._critic(data_train)

        return generator, critic

    def _generator(self, data_train: tuple[np.ndarray, np.ndarray]) -> Model:
        data_dimension = data_train[0].shape[1:]
        labels = data_train[1]

        hidden_dimension = 128

        model = Sequential([
            Dense(hidden_dimension, input_dim=self._noise_dimension),
            LeakyReLU(alpha=0.2),
            Dense(hidden_dimension * 2),
            LeakyReLU(alpha=0.2),
            Dense(hidden_dimension * 4),
            LeakyReLU(alpha=0.2),
            Dense(np.prod(data_dimension), activation='tanh'),
            Reshape(data_dimension)
        ])

        noise_input = Input(shape=self._noise_dimension)
        label_input = Input(shape=1)

        label_embedding = Embedding(max(labels) + 1, self._noise_dimension)(label_input)
        label_embedding = Flatten()(label_embedding)

        generator_input = multiply([noise_input, label_embedding])
        generator_output = model(generator_input)

        return Model([noise_input, label_input], generator_output, name=self._generator_name)

    def _critic(self, data_train: tuple[np.ndarray, np.ndarray]) -> Model:
        data_dimension = data_train[0].shape[1:]
        labels = data_train[1]

        hidden_dimension = 512

        model = Sequential([
            Dense(hidden_dimension, input_dim=np.prod(data_dimension)),
            LeakyReLU(alpha=0.2),
            Dense(hidden_dimension / 2),
            LeakyReLU(alpha=0.2),
            Dense(hidden_dimension / 4),
            LeakyReLU(alpha=0.2),
            Dense(1, activation='sigmoid')
        ])

        data_input = Input(shape=data_dimension)
        label_input = Input(shape=1)

        label_embedding = Embedding(max(labels) + 1, np.prod(data_dimension))(label_input)
        label_embedding = Flatten()(label_embedding)
        data_flatten = Flatten()(data_input)

        labeled_data_input = multiply([data_flatten, label_embedding])

        critic_validity = model(labeled_data_input)

        critic = Model([data_input, label_input], critic_validity, name=self._critic_name)

        return critic

    @staticmethod
    def _critic_loss(fake_logits, real_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    @staticmethod
    def _generator_loss(fake_logits):
        return -tf.reduce_mean(fake_logits)

    @staticmethod
    def _gradient_penalty(critic, data_real, data_fake, labels):
        data_shape_dimensions = [1] * len(data_real.shape[1:])
        epsilon = tf.random.uniform([data_real.shape[0], *data_shape_dimensions], minval=0, maxval=1)
        interpolated = data_real * epsilon + data_fake * (1 - epsilon)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            prediction = critic([interpolated, labels])

        axis = [index + one for index, one in enumerate(data_shape_dimensions)]
        gradients = tape.gradient(prediction, interpolated)
        gradient_l2_norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), axis=axis))
        gradient_penalty = tf.reduce_mean((gradient_l2_norm - 1.0) ** 2)

        return gradient_penalty

    @tf.function
    def _train_step(self, data_train, models, generator_optimizer, critic_optimizer):
        x_train, y_train = data_train
        x_train = tf.cast(x_train, dtype='float32')
        generator, critic = models

        for _ in range(self._critic_iterations):
            with tf.GradientTape() as critic_tape:
                noise_batch = tf.random.normal([x_train.shape[0], self._noise_dimension])

                generated_batch = generator([noise_batch, y_train], training=True)
                fake_logits = critic([generated_batch, y_train], training=True)
                real_logits = critic([x_train, y_train], training=True)

                critic_loss = self._critic_loss(fake_logits, real_logits)
                gradient_penalty = self._gradient_penalty(partial(critic, training=True),
                                                          x_train, generated_batch, y_train)
                critic_loss += self._gradient_penalty_weight * gradient_penalty

            critic_gradients = critic_tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        with tf.GradientTape() as generator_tape:
            noise_batch = tf.random.normal([x_train.shape[0], self._noise_dimension])
            generated_batch = generator([noise_batch, y_train], training=True)
            fake_logits = critic([generated_batch, y_train], training=True)
            generator_loss = self._generator_loss(fake_logits)

        generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        return critic_loss, generator_loss

    def _train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:

        data_train = (x_train, y_train)
        models = self._get_models(data_train)

        if self._loaded_pretrained_models:
            return None

        generator_optimizer = Adam(learning_rate=2e-6, beta_1=0, beta_2=0.9)
        critic_optimizer = RMSProp(learning_rate=5e-6)

        generator, critic = models
        history = History()
        history.model = generator
        history.on_train_begin()

        train_dataset = tf.data.Dataset.from_tensor_slices(data_train) \
            .shuffle(data_train[0].shape[0]).batch(self._batch_size)

        for epoch in tqdm(range(self._epochs)):
            critic_loss, generator_loss = 0, 0
            batches = 0
            for train_batch in train_dataset:
                batches += 1
                critic_loss_batch, generator_loss_batch = self._train_step(train_batch, models,
                                                                           generator_optimizer, critic_optimizer)

                critic_loss += critic_loss_batch
                generator_loss += generator_loss_batch

            critic_loss /= batches
            generator_loss /= batches

            history.on_epoch_end(epoch, {'Generator loss': generator_loss, 'Critic loss': critic_loss})

            if self._debug_save and epoch % self._debug_save_interval == 0:
                self._compare_generated(generator, data_train, epoch)
                visualization.plot_training_history(history, f"{epoch}_{self.get_name()}")

        file_manager.save_temporary_fold_model(generator, self._generator_fold_id)
        file_manager.save_temporary_fold_model(critic, self._critic_fold_id)
        file_manager.save_temporary_fold_history(history, self._generator_fold_id)

    def _compare_generated(self, generator: Model, data_train: tuple, epoch: int):
        x_train, y_train = data_train

        generated_data, generated_data_labels = self._generate_new_data(x_train.shape[0], np.unique(y_train),
                                                                        generator=generator)

        x_train_original = self._scaler.inverse_transform(x_train)
        visualization.plot_real_vs_generated((x_train_original, y_train), (generated_data, generated_data_labels),
                                             self.get_name(), f"{epoch}")

    def _plot_models(self, models: tuple) -> None:
        for model in models:
            visualization.save_model_graph(model, self.get_name())

    def _load_trained_models(self) -> tuple[Model, Model]:
        if self._loaded_pretrained_models:
            generator = file_manager.load_model(self.get_name(), self._generator_name)
            critic = file_manager.load_model(self.get_name(), self._critic_name)
        else:
            generator = file_manager.load_temporary_fold_model(self._generator_fold_id)
            critic = file_manager.load_temporary_fold_model(self._critic_fold_id)

        return generator, critic

    def _save_models(self, models: tuple[Model, Model]) -> None:
        if self._loaded_pretrained_models:
            return

        for model in models:
            file_manager.save_model(self.get_name(), model)

    def _generate_new_data(self, amount_to_generate: int, unique_labels: np.ndarray, generator: Model = None) \
            -> tuple[np.ndarray, np.ndarray]:
        amount_to_generate_per_label = int(amount_to_generate / len(unique_labels))

        generated_data = []
        generated_data_labels = []

        for label in unique_labels:
            noise = np.random.normal(size=(amount_to_generate_per_label, self._noise_dimension))
            noise_label = np.zeros(amount_to_generate_per_label) + label

            if generator is None:
                x_predicted = process_utils.model_predict([noise, noise_label], self._generator_fold_id,
                                                          remove=False)
            else:
                x_predicted = generator.predict([noise, noise_label], verbose=False)
            x_generated = self._scaler.inverse_transform(x_predicted)

            generated_data.append(x_generated)
            generated_data_labels.append(noise_label)

        return np.concatenate(generated_data), np.concatenate(generated_data_labels)

    def _remove_temporary_models(self) -> None:
        file_manager.remove_temporary_fold_model(self._generator_fold_id)
        file_manager.remove_temporary_fold_model(self._critic_fold_id)

    def get_name(self) -> str:
        return 'cWGAN-GP'
