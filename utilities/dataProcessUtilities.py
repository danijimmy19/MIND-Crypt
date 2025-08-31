"""
This script contains utility functions required for pre-processing the datasets.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# def convert_csv_to_numpy(csv_path, np_save_path, label):
#     """
#     This function is used for converting the csv files to the npz file.
#     :param csv_path: path to the csv file to be converted to npz file.
#     :param np_save_path: path to save the numpy array
#     :param label: train, valid, or test for labelling the set of the data
#     :return: None
#     """
#     print(f"converting the dataset for {label} set ...")
#     csv_df = pd.read_csv(csv_path, header=None)
#     print(csv_df.head())
#     print(f"dataset loaded successfully!")
#     csv_df.columns = ["message", "key", "iv", "ciphertext"]
#     print(f"shape of the df = {csv_df.shape}")
#     key_sample = csv_df["key"].iloc[0]
#     ivs = csv_df["iv"]
#     cts = csv_df["ciphertext"]
#
#     # print(f"saving the data to numpy array ...") 
#     # np.savez_compressed(np_save_path, iv=ivs, ciphertext=cts, keys=key_sample)
#     # print(f"numpy array containing csv file info saved successfully at: \n {csv_path}")
#     return np_save_path


def round_3_linear_attack_probability(plaintexts, ciphertexts, round_key_1, round_key_2, round_key_3):
    """
    This function is used for label assignment and computation of theoritical value for the linear attack on 3-round
    DES.
    :param plaintexts: The numpy array representing the plaintexts in binary.
    :param ciphertexts: The numpy array representing the ciphertexts in binary.
    :param roung_key_1: The numpy array representing the round key 1 in binary.
    :param round_key_2: The numpy array representing the round key 2 in binary.
    :param round_key_3: The numpy array representing the round key 3 in binary.
    :return: theoritical probability, features_set, label_set

    Input vector is represented as follows:
    [PH[7],PH[18],PH[24],PH[29],PL[15],CH[7],CH[18],CH[24],CH[29],CL[15]]
    """
    # Initialize a counter for matches
    matches = 0

    # To store vectors for each input-output pair
    vectors = []
    # labels corresponding to each vectors
    labels = []

    # Validate the equation for all data points
    for i in range(len(plaintexts)):
        PH = plaintexts[i][:32]  # First half of plaintext
        PL = plaintexts[i][32:]  # Second half of plaintext
        CH = ciphertexts[i][:32]  # First half of ciphertext
        CL = ciphertexts[i][32:]  # Second half of ciphertext

        # Correctly adjusting indexes based on reversed bit numbering
        # [7, 18, 24, 29]
        bit_positions_PH_CH = [24, 13, 7, 2]  # Adjusted for 0-based indexing in 32 bits
        key_bit_position = 25  # Adjusted position for key bits in 48 bits

        # Extracting specific bits and performing XOR
        equation_left = PL[16] ^ CL[16]  # Correcting the indexing for PL and CL
        for pos in bit_positions_PH_CH:
            equation_left ^= PH[pos] ^ CH[pos]

        vector = [PH[24], PH[13], PH[7], PH[2], PL[16],
                  CH[24], CH[13], CH[7], CH[2], CL[16]]

        equation_right = round_key_1[i][key_bit_position] ^ round_key_3[i][key_bit_position]

        # Increment the counter if the equation holds
        if equation_left == equation_right:
            matches += 1
            labels.append(1)
        else:
            labels.append(0)

        # Append the current vector to the list of vectors
        vectors.append(vector)

    # Calculate the empirical probability
    probability = matches / len(plaintexts)
    print(f"The empirical probability that the equation holds is approximately {probability:.5f}.")

    # Theoritical probability
    theoretical_probability = (12 / 64) ** 2 + (1 - 12 / 64) ** 2
    print(f"The theoretical probability is {theoretical_probability:.2f}.")

    return {"empirical_probability": probability,
            "theoretical_probability": theoretical_probability,
            "features": vectors, "labels": labels}


def prepare_round_3_linear_des_dataset(dataset_path, column_names, set_label=None):
    """
    This function is used for preparing the dataset linear attacks on DES. More specifically, the hex representation
    of the dataset is converted to binary representation.
    :param dataset_path: path to the CSV file containing hex representation of the dataset.
    :param column_names: the column names of the dataset.
    :param set_label: "training", "validation", or "testing" identification of the label
    :return: path of the numpy array where the processed dataset is stored.
    """
    print(f"processing the {set_label} dataset ...")
    data_df = pd.read_csv(dataset_path)
    data_df = data_df.head(10**7)
    data_df.columns = column_names
    print(f"shape of the dataset = {data_df.shape}")

    # Vectorize your function
    v_convert_to_binary = np.vectorize(convert_to_binary)

    # Convert arrays
    print(f"converting the dataset to binary vectors ...")
    ciphertexts = v_convert_to_binary(data_df["ciphertext"].values)
    round_key_1 = v_convert_to_binary(data_df["round_key_1"].values)
    round_key_2 = v_convert_to_binary(data_df["round_key_2"].values)
    round_key_3 = v_convert_to_binary(data_df["round_key_3"].values)

    ciphertexts = np.array([list(map(int, list(binary_string))) for binary_string in ciphertexts], dtype=np.uint16)
    round_key_1 = np.array([list(map(int, list(binary_string))) for binary_string in round_key_1], dtype=np.uint16)
    round_key_2 = np.array([list(map(int, list(binary_string))) for binary_string in round_key_2], dtype=np.uint16)
    round_key_3 = np.array([list(map(int, list(binary_string))) for binary_string in round_key_3], dtype=np.uint16)

    plaintexts = v_convert_to_binary(data_df["message"].values)
    plaintexts = np.array([list(map(int, list(binary_string))) for binary_string in plaintexts], dtype=np.uint16)

    print("looking at shapes of different arrays ...")
    print(f"plaintexts = {plaintexts.shape}")
    print(f"ciphertexts = {ciphertexts.shape}")
    print(f"round_key_1 = {round_key_1.shape}")
    print(f"round_key_2 = {round_key_2.shape}")
    print(f"round_key_3 = {round_key_3.shape}")

    extract_features = round_3_linear_attack_probability(plaintexts, ciphertexts, round_key_1, round_key_2, round_key_3)

    print(f"processing the {set_label} dataset completed successfully!")

    return extract_features


def convert_to_binary(hex_string):
    """
    This is a helper function used to convert hex string to binary.
    :param hex_string: String to be converted to binary
    :return: binary string
    """
    bin_representation = bin(int(hex_string, 16))[2:].zfill(len(hex_string)*4)
    return bin_representation


def convert_to_hex(binary_string):
    """
    This is a helper function used to convert binary to hex.
    :param binary_string: A binary string to be converted to hex
    :return: String converted to hex
    """
    hex_string = format(int(binary_string, 2), 'x').zfill((len(binary_string) + 3) // 4)
    return hex_string


def prepare_dataset(np_array_path, n_samples_per_class):
    """
    This function is used for preparing the dataset for model input generation.
    :param np_array_path: A path to the .npy file
    :param n_samples_per_class: Number of samples to be considered per class
    :return: A numpy array containing the binary representation of the cipher of the message
    """
    np_ciphertexts = np.load(np_array_path, allow_pickle=True)
    ciphertexts = np_ciphertexts["ciphertext"]
    print(f"shape of the ciphertexts: {ciphertexts.shape}")

    binary_representation = []
    for i in tqdm(range(n_samples_per_class), desc="converting to binary ..."):
        binary_string = convert_to_binary(ciphertexts[i])
        binary_string_list = [int(bit) for bit in binary_string]
        binary_representation.append(binary_string_list)

    return np.array(binary_representation), np_ciphertexts["keys"]


def merge_data(ciphertexts_1, labels_1, ciphertexts_2, labels_2):
    """
    This function is used for merging the two numpy arrays
    :param ciphertexts_1: A numpy array containing binary representation of ciphertext 1
    :param labels_1: A numpy array containing labels of ciphertext 1
    :param ciphertexts_2: A numpy array containing binary representation of ciphertext 2
    :param labels_2: A numpy array containing labels of ciphertext 1
    :return: Two numpy arrays containing input and labels for the NN training
    """
    x = np.concatenate([ciphertexts_1, ciphertexts_2], axis=0)
    y = np.concatenate([labels_1, labels_2], axis=0)

    # Generate a list of shuffled indices
    shuffled_indices = np.arange(x.shape[0])
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to reorder both X and Y
    x = x[shuffled_indices]
    y = y[shuffled_indices]

    return x, y


def convert_csv_to_numpy(csv_path, np_save_path):
    """
    This function is used for converting the csv files to the npz file.
    :param csv_path: path to the csv file to be converted to npz file.
    :param np_save_path: path to save the numpy array
    :return: None
    """
    return None