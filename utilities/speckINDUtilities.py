"""
This script contains all the utility functions written by Gohr for performing encryption and decryption operations
on the data using SPECK32/64 ciphers.
"""

import numpy as np
from os import urandom


def WORD_SIZE():
    return (16);


def ALPHA():
    return (7);


def BETA():
    return (2);


MASK_VAL = 2 ** WORD_SIZE() - 1;


def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);


def rol(x, k):
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));


def ror(x, k):
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));


def enc_one_round(p, k):
    c0, c1 = p[0], p[1];  # p[0] = l[i%3], p[1] = ks[i], k = ith round
    c0 = ror(c0, ALPHA());  # Li >> ALPHA --> Li_RotR
    c0 = (c0 + c1) & MASK_VAL;  # (Li_RotR square+ Ri) --> (modulo addition followed by bitwise &) square+ -- Modular
    # addition modulo 2^n
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return (c0, c1);


def dec_one_round(c, k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return (c0, c1);


def expand_key(k, t):
    """
    :param k: keys -> [[key1], [key2], ..., [keyn]] where key_i = key per data
    :param t: Number of rounds
    """
    ks = [0 for i in range(t)];
    ks[0] = k[len(k) - 1];
    l = list(reversed(k[:len(k) - 1]));
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i);

    return (ks);


def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x, y = enc_one_round((x, y), k);
    return (x, y);


def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x, y), k);
    return (x, y);


def check_testvector():
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    if (ct == (0xa868, 0x42f2)):
        print("Testvector verified.")
        return (True);
    else:
        print("Testvector not verified.")
        return (False);


# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
    X = np.zeros((2 * WORD_SIZE(), len(arr[0])), dtype=np.uint8);
    for i in range(2 * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return (X);


# takes a text file that contains encrypted block0, block1, true diff prob, real or random
# data samples are line separated, the above items whitespace-separated
# returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s, 16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16);
    ct1a = np.array(ct1a, dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16);
    ct1b = np.array(ct1b, dtype=np.uint16);

    # X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]);
    Y = np.array(Y, dtype=np.uint8);
    Z = np.array(Z);
    return (X, Y, Z);


# baseline training data generator
def make_train_data_enc_0_vs_enc_1(n, nr, keys):
    """
    The function generates training data where each plaintext message is XORed with an
    Initialization Vector (IV) to add randomization. Each plaintext set is assigned a
    fixed label: Y=0 for the first set (plain0*) and Y=1 for the second set (plain1*),
    with additional data for Enc(0) and Enc(1).
    """
    print("Generating data for SPECK32/64 CBC with Enc(0) and Enc(1)...")

    # Initial fixed plaintext values
    plain0l = np.zeros(n, dtype=np.uint16)
    plain0r = np.zeros(n, dtype=np.uint16)
    plain1l = np.zeros(n, dtype=np.uint16)  # Assuming intention for Enc(0)
    plain1r = np.ones(n, dtype=np.uint16)  # Assuming intention for Enc(1)

    ks = expand_key(keys, nr)

    # Generating random IVs for each plaintext message
    iv1 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    iv2 = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # For demonstration, print the binary representation of the first few elements
    few = min(n, 5)  # Adjust to change how many elements to print
    for i in range(few):
        print(f"iv1[{i}]: {iv1[i]}")
        print(f"plain0l[{i}]: {format(plain0l[i], '016b')}, plain0r[{i}]: {format(plain0r[i], '016b')}")
        print(f"iv2[{i}]: {iv2[i]}")
        print(f"plain1l[{i}]: {format(plain1l[i], '016b')}, plain1r[{i}]: {format(plain1r[i], '016b')}")
        print("-" * 99)

    # XOR plaintexts with IVs before encryption
    ctdata0l, ctdata0r = encrypt((plain0l ^ iv1, plain0r ^ iv1), ks)
    ctdata1l, ctdata1r = encrypt((plain1l ^ iv2, plain1r ^ iv2), ks)

    # Convert ciphertexts to binary format
    X0 = convert_to_binary([ctdata0l, ctdata0r])
    X1 = convert_to_binary([ctdata1l, ctdata1r])
    print(f"X0 = {X0}")
    print(f"X1 = {X1}")

    # Since labels are fixed, no need to generate Y as before
    Y0 = np.zeros(n, dtype=np.uint8)  # Label for the first message set
    Y1 = np.ones(n, dtype=np.uint8)  # Label for the second message set

    # Combine the data and labels
    X = np.concatenate([X0, X1], axis=0)
    Y = np.concatenate([Y0, Y1], axis=0)

    # Generate a list of shuffled indices
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to reorder both X and Y
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]

    # Decryption process
    decrypted0l, decrypted0r = decrypt((ctdata0l, ctdata0r), ks)
    decrypted1l, decrypted1r = decrypt((ctdata1l, ctdata1r), ks)

    # XOR with IV to get back the original plaintexts
    decrypted0l ^= iv1;
    decrypted0r ^= iv1
    decrypted1l ^= iv2;
    decrypted1r ^= iv2

    print(f"Decrypted (plain0l, plain0r) = ({decrypted0l[:5]}, {decrypted0r[:5]})")
    print(f"Decrypted (plain1l, plain1r) = ({decrypted1l[:5]}, {decrypted1r[:5]})")

    return X, Y


# # Training Data Generator (Updated)
# def make_train_data_enc_0_vs_enc_1(num_of_samples, num_of_rounds, block_size, num_of_key_words, keys):
#     """
#     The function generates training data where each plaintext message is XORed with an
#     Initialization Vector (IV) to add randomization. Each plaintext set is assigned a
#     fixed label: Y=0 for the first set (plain0) and Y=1 for the second set (plain1),
#     with additional data for Enc(0) and Enc(1).
#     """
#     # Based on Block size, set the appropriate vars
#     if block_size == 32:
#         data_type = np.uint16
#         print_pattern = '016b'
#
#     elif block_size == 64:
#         data_type = np.uint32
#         print_pattern = '032b'
#
#     elif block_size == 128:
#         data_type = np.uint64
#         print_pattern = '064b'
#
#     multiplier = block_size // 8
#
#     print("Generating data for SPECK32/64 CBC with Enc(0) and Enc(1)...")
#
#     # Initial fixed plaintext values
#     # plain0l = np.zeros(n, dtype=np.uint16)
#     # plain0r = np.zeros(n, dtype=np.uint16)
#     # plain1l = np.zeros(n, dtype=np.uint16)  # Assuming intention for Enc(0)
#     # plain1r = np.ones(n, dtype=np.uint16)  # Assuming intention for Enc(1)
#     plain0l = np.zeros(multiplier * num_of_samples, dtype=data_type)
#     plain0r = np.zeros(multiplier * num_of_samples, dtype=data_type)
#     plain1l = np.zeros(multiplier * num_of_samples, dtype=data_type)  # Assuming intention for Enc(0)
#     plain1r = np.ones(multiplier * num_of_samples, dtype=data_type)  # Assuming intention for Enc(1)
#
#     # Generation of Keys (if needed)
#     # keys = np.frombuffer(urandom(num_of_key_words  multiplier  num_of_samples),dtype=data_type).reshape(num_of_key_words,-1)
#
#     ks = expand_key(keys, num_of_rounds)
#
#     # Generating random IVs for each plaintext message
#     iv1 = np.frombuffer(urandom(multiplier * num_of_samples), dtype=data_type)
#     iv2 = np.frombuffer(urandom(multiplier * num_of_samples), dtype=data_type)
#
#     # For demonstration, print the binary representation of the first few elements
#     few = min(num_of_samples, 5)  # Adjust to change how many elements to print
#     for i in range(few):
#         print(f"plain0l[{i}]: {format(plain0l[i], print_pattern)}, plain0r[{i}]: {format(plain0r[i], print_pattern)}")
#         print(f"plain1l[{i}]: {format(plain1l[i], print_pattern)}, plain1r[{i}]: {format(plain1r[i], print_pattern)}")
#         print("-" * 99)
#
#     # XOR plaintexts with IVs before encryption
#     ctdata0l, ctdata0r = encrypt((plain0l ^ iv1, plain0r ^ iv1), ks)
#     ctdata1l, ctdata1r = encrypt((plain1l ^ iv2, plain1r ^ iv2), ks)
#
#     # Convert ciphertexts to binary format
#     X0 = convert_to_binary([ctdata0l, ctdata0r])
#     X1 = convert_to_binary([ctdata1l, ctdata1r])
#     print(f"X0 = {X0}")
#     print(f"X1 = {X1}")
#
#     # Since labels are fixed, no need to generate Y as before
#     Y0 = np.zeros(num_of_samples, dtype=np.uint8)  # Label for the first message set
#     Y1 = np.ones(num_of_samples, dtype=np.uint8)  # Label for the second message set
#
#     # Combine the data and labels
#     X = np.concatenate([X0, X1], axis=0)
#     Y = np.concatenate([Y0, Y1], axis=0)
#
#     # Decryption process
#     decrypted0l, decrypted0r = decrypt((ctdata0l, ctdata0r), ks)
#     decrypted1l, decrypted1r = decrypt((ctdata1l, ctdata1r), ks)
#
#     # XOR with IV to get back the original plaintexts
#     decrypted0l ^= iv1
#     decrypted0r ^= iv1
#     decrypted1l ^= iv2
#     decrypted1r ^= iv2
#
#     print(f"Decrypted (plain0l, plain0r) = ({decrypted0l[:5]}, {decrypted0r[:5]})")
#     print(f"Decrypted (plain1l, plain1r) = ({decrypted1l[:5]}, {decrypted1r[:5]})")
#
#     return X, Y
