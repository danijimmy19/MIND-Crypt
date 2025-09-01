"""
This script is used for generating the dataset for SINON32/64 cipher for the indistinguishability experiment.
"""
import os
from simon import SimonCipher as simon
import numpy as np
import ctypes

def to_int32(bit):
    """Converts a single-bit integer to a 32-bit integer."""
    return ctypes.c_int32(bit).value

# path to save the output file
data_path = "../../../../../../data-simon-32-64-enc-0-vs-enc-1-rounds-5/"
os.makedirs(data_path, exist_ok=True)
data_file = os.path.join(data_path, "train-data.npz")

# Key to be used for encryption (64-bit)
key=0x11223344
# key=0x112233445566778899aabbccddeeff00

# two plaintext messages
messages = [0, 1]

# Number of samples to collect for each message
n = (10 ** 7) // 2

# To save the generated dataset
plain_text = []
cipher_text = []
cipher_text_binary = []

# modes of operation of cipher
# cipher_modes = [ 'CTR', 'CBC', 'PCBC', 'CFB', 'OFB' ]
cipher_modes = ['CBC']

# Evaluate single-bit integers 0 and 1:
for single_bit_int in messages:
    print(f"processing message {single_bit_int}")
    # Convert to 32-bit integer using your function
    my_plaintext = to_int32(single_bit_int)

    for mode in cipher_modes:
        # print(f'-- Mode {mode} --')

        # Instantiate encrypt/decrypt for this mode
        E = simon(key, key_size=64, block_size=32, mode=mode).encrypt
        D = simon(key, key_size=64, block_size=32, mode=mode).decrypt

        # Collect n samples for this plaintext and mode
        for i in range(n):
            # Encrypt
            encrypted_text = E(my_plaintext)

            # Convert ciphertext to binary string (adjust bit-length if needed)
            encrypted_binary = format(encrypted_text, '032b')

            # Optional: Decrypt to confirm correctness
            decrypted_text = D(encrypted_text)

            # Print if you want to see each sample
            # print(f"Sample {i + 1}:")
            # print(f"  Plaintext (decimal) = {single_bit_int}")
            # print(f"  Ciphertext (decimal) = {encrypted_text}")
            # print(f"  Ciphertext (binary)  = {encrypted_binary}")
            # print(f"  Ciphertext (binary)  = {len(encrypted_binary)}")
            # print(f"  Decrypted text       = {decrypted_text}")
            # print()

            # Append to results list
            # Storing (plaintext, ciphertext_decimal, ciphertext_binary, mode)
            plain_text.append(my_plaintext)
            cipher_text.append(encrypted_text)
            cipher_text_binary.append(encrypted_binary)
            # results.append([single_bit_int, encrypted_text, encrypted_binary, mode])


print(f"unique messages = {len(set(plain_text))}")
print(f"unique cipher_text = {len(set(cipher_text_binary))}")
print(f"unique cipher_text_binary = {len(set(cipher_text_binary))}")

# save the generated dataset
np.savez(data_file, key=key,
         plain_text=plain_text,
         cipher_text=cipher_text,
         cipher_text_binary=cipher_text_binary)