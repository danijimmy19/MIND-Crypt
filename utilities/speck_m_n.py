import numpy as np
from os import urandom


class Speck:
    def __init__(self, block_size, key_size, keys=None) -> None:
        self.__block_size = block_size
        self.__key_size = key_size
        self.__word_length = self.__block_size // 2
        self.__num_of_key_words = self.__key_size // self.__word_length
        self.__mask_value = 2 ** self.__word_length - 1

        # Fix the values of alpha & beta
        # using block size
        if self.__block_size == 32:
            self.__alpha = 7
            self.__beta = 2
        else:
            self.__alpha = 8
            self.__beta = 3

        # Fix the data type using word length
        if self.__word_length == 16:
            self.data_type = np.uint16

        elif self.__word_length == 32:
            self.data_type = np.uint32

        elif self.__word_length == 64:
            self.data_type = np.uint64

        self.__keys = keys if keys else None

    def generate_keys(self):
        self.__keys = np.frombuffer(urandom(self.__num_of_key_words * (self.__word_length // 8)),
                                    dtype=self.data_type).reshape(self.__num_of_key_words, -1)

    def get_keys(self):
        return self.__keys

    def get_alpha(self):
        return self.__alpha

    def get_beta(self):
        return self.__beta

    def get_mask_value(self):
        return self.__mask_value

    def get_word_size(self):
        return self.__word_length

    def get_block_size(self):
        return self.__block_size

    def get_key_size(self):
        return self.__key_size

    def get_num_of_key_words(self):
        return self.__num_of_key_words

    def rotate_to_left(self, txt, k_val):
        return (((txt << k_val) & self.get_mask_value()) | (txt >> (self.get_word_size() - k_val)))

    def rotate_to_right(self, txt, k_val):
        return ((txt >> k_val) | ((txt << (self.get_word_size() - k_val)) & self.get_mask_value()))

    def enc_one_round(self, plain_txt, round_num):
        cipher_txt0, cipher_txt1 = plain_txt[0], plain_txt[1]
        cipher_txt0 = self.rotate_to_right(txt=cipher_txt0, k_val=self.get_alpha())
        cipher_txt0 = (cipher_txt0 + cipher_txt1) & self.get_mask_value()
        cipher_txt0 = cipher_txt0 ^ round_num
        cipher_txt1 = self.rotate_to_left(txt=cipher_txt1, k_val=self.get_beta())
        cipher_txt1 = cipher_txt1 ^ cipher_txt0
        return (cipher_txt0, cipher_txt1)

    def dec_one_round(self, cipher_txt, round_num):
        cipher_txt0, cipher_txt1 = cipher_txt[0], cipher_txt[1]
        cipher_txt1 = cipher_txt1 ^ cipher_txt0
        cipher_txt1 = self.rotate_to_right(txt=cipher_txt1, k_val=self.get_beta())
        cipher_txt0 = cipher_txt0 ^ round_num
        cipher_txt0 = (cipher_txt0 - cipher_txt1) & self.get_mask_value()
        cipher_txt0 = self.rotate_to_left(txt=cipher_txt0, k_val=self.get_alpha())
        return (cipher_txt0, cipher_txt1)

    def expand_key(self, num_of_rounds):
        keys = self.get_keys()

        exp_keys = [0 for i in range(num_of_rounds)]
        exp_keys[0] = keys[len(keys) - 1]

        helper_list = list(reversed(keys[:len(keys) - 1]))

        for round_i in range(num_of_rounds - 1):
            helper_ind = round_i % (self.get_num_of_key_words() - 1)
            helper_list[helper_ind], exp_keys[round_i + 1] = self.enc_one_round(
                (helper_list[helper_ind], exp_keys[round_i]), round_i)

        return exp_keys

    def encrypt(self, plain_txt, exp_keys):
        left_word, right_word = plain_txt[0], plain_txt[1]
        for exp_key in exp_keys:
            left_word, right_word = self.enc_one_round((left_word, right_word), exp_key)
        return (left_word, right_word)

    def decrypt(self, cipher_txt, exp_keys):
        left_word, right_word = cipher_txt[0], cipher_txt[1]
        for exp_key in reversed(exp_keys):
            left_word, right_word = self.dec_one_round((left_word, right_word), exp_key)
        return (left_word, right_word)

    def convert_to_binary(self, arr):
        ret_x = np.zeros((2 * self.get_word_size(), len(arr[0])), dtype=np.uint8)
        for i in range(2 * self.get_word_size()):
            index = i // self.get_word_size()
            offset = self.get_word_size() - (i % self.get_word_size()) - 1
            ret_x[i] = (arr[index] >> offset) & 1

        ret_x = ret_x.transpose()
        return ret_x


def make_train_data_enc_0_vs_1(num_of_samples, num_of_rounds, speck):
    # Calculate multiplier based on Speck's
    # Block size
    multiplier = speck.get_block_size() // 16

    print(f"Data type: {speck.data_type} and Multiplier: {multiplier}")
    print(f"Generating data for SPECK{speck.get_block_size()}/{speck.get_key_size()} CBC with Enc(0) and Enc(1)...")

    plain0l = np.zeros(num_of_samples, dtype=speck.data_type)
    plain0r = np.zeros(num_of_samples, dtype=speck.data_type)
    plain1l = np.zeros(num_of_samples, dtype=speck.data_type)  # Assuming intention for Enc(0)
    plain1r = np.ones(num_of_samples, dtype=speck.data_type)  # Assuming intention for Enc(1)

    ks = speck.expand_key(num_of_rounds)

    # Generating random IVs for each plaintext message
    iv1 = np.frombuffer(urandom(multiplier * num_of_samples), dtype=speck.data_type)
    iv2 = np.frombuffer(urandom(multiplier * num_of_samples), dtype=speck.data_type)

    few = min(num_of_samples, 5)  # Adjust to change how many elements to print
    for i in range(few):
        print(f"plain0l[{i}]: {plain0l[i]}, plain0r[{i}]: {plain0r[i]}")
        print(f"plain1l[{i}]: {plain1l[i]}, plain1r[{i}]: {plain1r[i]}")
        print("-" * 99)

    # XOR plaintexts with IVs before encryption
    ctdata0l, ctdata0r = speck.encrypt((plain0l ^ iv1, plain0r ^ iv1), ks)
    ctdata1l, ctdata1r = speck.encrypt((plain1l ^ iv2, plain1r ^ iv2), ks)

    for i in range(few):
        print(f"ctdata0l[{i}]: {ctdata0l[i]}, ctdata0r[{i}]: {ctdata0r[i]}")
        print(f"ctdata1l[{i}]: {ctdata1l[i]}, ctdata1r[{i}]: {ctdata1r[i]}")
        print("-" * 99)

    # Convert ciphertexts to binary format
    X0 = speck.convert_to_binary([ctdata0l, ctdata0r])
    X1 = speck.convert_to_binary([ctdata1l, ctdata1r])

    for i in range(few):
        print(f"X0[{i}] = {X0[i]}, Length = {len(X0[i])}")
        print(f"X1[{i}] = {X1[i]}, Length = {len(X1[i])}")
        print("-" * 99)

    # Since labels are fixed, no need to generate Y as before
    Y0 = np.zeros(num_of_samples, dtype=np.uint8)  # Label for the first message set
    Y1 = np.ones(num_of_samples, dtype=np.uint8)  # Label for the second message set

    # Combine the data and labels
    X = np.concatenate([X0, X1], axis=0)
    Y = np.concatenate([Y0, Y1], axis=0)

    # Generate a list of shuffled indices
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to reorder both X and Y
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]

    ###################
    # Sanity Checking #
    ###################

    # Decryption process
    decrypted0l, decrypted0r = speck.decrypt((ctdata0l, ctdata0r), ks)
    decrypted1l, decrypted1r = speck.decrypt((ctdata1l, ctdata1r), ks)

    # XOR with IV to get back the original plaintexts
    decrypted0l ^= iv1
    decrypted0r ^= iv1
    decrypted1l ^= iv2
    decrypted1r ^= iv2

    for i in range(few):
        print(f"Decrypted ==> plain0l[{i}]: {decrypted0l[i]}, plain0r[{i}]: {decrypted0r[i]}")
        print(f"Decrypted ==> plain1l[{i}]: {decrypted1l[i]}, plain1r[{i}]: {decrypted1r[i]}")
        print("-" * 99)

    return X, Y


# if __name__ == "__main__":
#     # Create instance of Speck and
#     # generate keys
#     speck = Speck(block_size=32, key_size=64)
#     speck.generate_keys()
#
#     # Call make_train_data_enc_0_vs_1 function
#     # passing 'Number of Samples', 'Number of Rounds
#     # and Speck instance
#     X, Y = make_train_data_enc_0_vs_1(num_of_samples=1, num_of_rounds=8, speck=speck)