import functools
import operator
import os
import time

import numpy as np
import config
import model


class VideoDescriptionInference(object):
    """
            Initialize the parameters for the model
            """
    def __init__(self, config):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability

        # models
        self.tokenizer, self.inf_encoder_model,  self.inf_decoder_model = model.inference_model()
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        self.search_type = config.search_type

    def greedy_search(self, loaded_array):
        """

                :param f: the loaded numpy array after creating videos to frames and extracting features
                :return: the final sentence which has been predicted greedily
                """
        inv_map = self.index_to_word()
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, 1500))
        sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)
            if y_hat == 0:
                continue
            if inv_map[y_hat] is None:
                break
            else:
                sentence = sentence + inv_map[y_hat] + ' '
                target_seq = np.zeros((1, 1, 1500))
                target_seq[0, 0, y_hat] = 1
        return ' '.join(sentence.split()[:-1])

    def decode_sequence2bs(self, input_seq):
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value, [], [], 0)
        return decode_seq

    def beam_search(self, target_seq, states_value, prob, path, lens):
        """

                :param target_seq: the array that is fed into the model to predict the next word
                :param states_value: previous state that is fed into the lstm cell
                :param prob: probability of predicting a word
                :param path: list of words from each sentence
                :param lens: number of words
                :return: final sentence
                """
        global decode_seq
        node = 2
        output_tokens, h, c = self.inf_decoder_model.predict(
            [target_seq] + states_value)
        output_tokens = output_tokens.reshape(self.num_decoder_tokens)
        sampled_token_index = output_tokens.argsort()[-node:][::-1]
        states_value = [h, c]
        for i in range(node):
            if sampled_token_index[i] == 0:
                sampled_char = ''
            else:
                sampled_char = list(self.tokenizer.word_index.keys())[
                    list(self.tokenizer.word_index.values()).index(sampled_token_index[i])]
            MAX_LEN = 12
            if sampled_char != 'eos' and lens <= MAX_LEN:
                p = output_tokens[sampled_token_index[i]]
                if sampled_char == '':
                    p = 1
                prob_new = list(prob)
                prob_new.append(p)
                path_new = list(path)
                path_new.append(sampled_char)
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index[i]] = 1.
                self.beam_search(target_seq, states_value, prob_new, path_new, lens + 1)
            else:
                p = output_tokens[sampled_token_index[i]]
                prob_new = list(prob)
                prob_new.append(p)
                p = functools.reduce(operator.mul, prob_new, 1)
                if p > self.max_probability:
                    decode_seq = path
                    self.max_probability = p

    def decoded_sentence_tuning(self, decoded_sentence):
        decode_str = []
        filter_string = ['bos', 'eos']
        uni_gram = {}
        last_string = ""
        for idx2, c in enumerate(decoded_sentence):
            if c in uni_gram:
                uni_gram[c] += 1
            else:
                uni_gram[c] = 1
            if last_string == c and idx2 > 0:
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if idx2 > 0:
                last_string = c
        return decode_str

    def index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word

    def get_test_data(self):
        """
        loads all the numpy files
        :return: two lists containing all the video arrays and the video Id
        """
        X_test = []
        X_test_filename = []
        with open(os.path.join(self.test_path, 'testing_id.txt')) as testing_file:
            lines = testing_file.readlines()
            for filename in lines:
                filename = filename.strip()
                f = np.load(os.path.join(self.test_path, 'feat', filename + '.npy'))
                X_test.append(f)
                X_test_filename.append(filename[:-4])
            X_test = np.array(X_test)
        return X_test, X_test_filename

    def test(self):
        """
            writes the captions of all the testing videos in a text file
        """
        X_test, X_test_filename = self.get_test_data()

        # generate inference test outputs
        with open(os.path.join(self.test_path, 'test_%s.txt' % self.search_type), 'w') as file:
            for idx, x in enumerate(X_test):
                file.write(X_test_filename[idx] + ',')
                if self.search_type is 'greedy':
                    start = time.time()
                    decoded_sentence = self.greedy_search(x.reshape(-1, 80, 4096))
                    file.write(decoded_sentence + ',{:.2f}'.format(time.time()-start))
                else:
                    start = time.time()
                    decoded_sentence = self.decode_sequence2bs(x.reshape(-1, 80, 4096))
                    decode_str = self.decoded_sentence_tuning(decoded_sentence)
                    for d in decode_str:
                        file.write(d + ' ')
                    file.write(',{:.2f}'.format(time.time() - start))
                file.write('\n')

                # re-init max prob
                self.max_probability = -1


if __name__ == "__main__":
    video_to_text = VideoDescriptionInference(config)
    video_to_text.test()
