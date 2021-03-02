import json
import os
import random

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import joblib
import config


class VideoDescriptionTrain(object):
    """
    Initialize the parameters for the model
    """

    def __init__(self, config):
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.epochs = config.epochs
        self.latent_dim = config.latent_dim
        self.validation_split = config.validation_split
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.time_steps_decoder = None
        self.train_list = []
        self.x_data = {}

        # processed data
        self.encoder_input_data = []
        self.decoder_input_data = []
        self.decoder_target_data = []
        self.tokenizer = None

        # models
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path

    def preprocessing(self):
        """
        Preprocessing the data
        dumps values of the json file into a list
        """
        TRAIN_LABEL_PATH = os.path.join(self.train_path, 'training_label.json')
        with open(TRAIN_LABEL_PATH) as data_file:
            y_data = json.load(data_file)
        train_list = []
        vocab_list = []
        for y in y_data:
            for caption in y['caption']:
                caption = "<bos> " + caption + " <eos>"
                if len(caption.split()) > 10 or len(caption.split()) < 6:
                    continue
                else:
                    train_list.append([caption, y['id']])

        random.shuffle(train_list)
        training_list = train_list[int(len(train_list) * self.validation_split):]
        validation_list = train_list[:int(len(train_list) * self.validation_split)]
        for train in training_list:
            vocab_list.append(train[0])
        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        self.tokenizer.fit_on_texts(vocab_list)

        TRAIN_FEATURE_DIR = os.path.join(self.train_path, 'feat')
        for filename in os.listdir(TRAIN_FEATURE_DIR):
            f = np.load(os.path.join(TRAIN_FEATURE_DIR, filename), allow_pickle=True)
            self.x_data[filename[:-4]] = f
        return training_list, validation_list

    def load_dataset(self, training_list):
        """
        Loads the dataset in batches for training
        :return: batch of data
        """
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []
        videoId = []
        videoSeq = []
        for idx, cap in enumerate(training_list):
            caption = cap[0]
            videoId.append(cap[1])
            videoSeq.append(caption)
        train_sequences = self.tokenizer.texts_to_sequences(videoSeq)
        train_sequences = np.array(train_sequences)
        train_sequences = pad_sequences(train_sequences, padding='post', truncating='post',
                                        maxlen=self.max_length)
        file_size = len(train_sequences)
        n = 0
        for i in range(self.epochs):
            for idx in range(0, file_size):
                n += 1
                encoder_input_data.append(self.x_data[videoId[idx]])
                y = to_categorical(train_sequences[idx], self.num_decoder_tokens)
                decoder_input_data.append(y[:-1])
                decoder_target_data.append(y[1:])
                if n == self.batch_size:
                    encoder_input = np.array(encoder_input_data)
                    decoder_input = np.array(decoder_input_data)
                    decoder_target = np.array(decoder_target_data)
                    encoder_input_data = []
                    decoder_input_data = []
                    decoder_target_data = []
                    n = 0
                    yield ([encoder_input, decoder_input], decoder_target)

    def train_model(self):
        """
        an encoder decoder sequence to sequence model
        reference : https://arxiv.org/abs/1505.00487
        """
        encoder_inputs = Input(shape=(self.time_steps_encoder, self.num_encoder_tokens), name="encoder_inputs")
        encoder = LSTM(self.latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(self.time_steps_decoder, self.num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='relu', name='decoder_relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # model.summary()
        training_list, validation_list = self.preprocessing()

        train = self.load_dataset(training_list)
        valid = self.load_dataset(validation_list)

        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')

        # Run training
        opt = keras.optimizers.Adam(lr=0.0003)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                      factor=0.1, patience=5, verbose=0,
                                                      mode="auto")
        model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')

        validation_steps = len(validation_list)//self.batch_size
        steps_per_epoch = len(training_list)//self.batch_size

        model.fit(train, validation_data=valid, validation_steps=validation_steps,
                  epochs=self.epochs, steps_per_epoch=steps_per_epoch,
                  callbacks=[reduce_lr, early_stopping])

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        # self.encoder_model.summary()
        # self.decoder_model.summary()

        # saving the models
        self.encoder_model.save(os.path.join(self.save_model_path, 'encoder_model.h5'))
        self.decoder_model.save_weights(os.path.join(self.save_model_path, 'decoder_model_weights.h5'))
        with open(os.path.join(self.save_model_path, 'tokenizer' + str(self.num_decoder_tokens)), 'wb') as file:
            joblib.dump(self.tokenizer, file)


if __name__ == "__main__":
    video_to_text = VideoDescriptionTrain(config)
    video_to_text.train_model()
