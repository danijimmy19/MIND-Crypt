"""
This is a prototype code for generating the dataset for simon32/64 cipher.
"""

from simon import SimonCipher as simon

import ctypes

def to_int32(bit):
    """Converts a single-bit integer to a 32-bit integer."""
    return ctypes.c_int32(bit).value

def int2bar(i, n): # convert int i to array of n bytes, little endian first
    return bytes([i>>(j<<3) & 0xff for j in range(0,n)])

def bar2int(bar): # convert byte array to int
    return sum([b << (j<<3) for j,b in enumerate(bar)])

key=0x112233445566778899aabbccddeeff00

# msg=b"Is it possible for you to provide one example of how to use 'CTR', 'CBC', 'PCBC', 'CFB', 'OFB' modes for both encryption and decryption?"
single_bit_int = 0
my_plaintext = to_int32(single_bit_int)
print(f"The 32-bit integer representation of {single_bit_int} is: {my_plaintext}")

# cipher_modes = [ 'CTR', 'CBC', 'PCBC', 'CFB', 'OFB' ]
cipher_modes = ['CBC']

for mode in cipher_modes:
    print('-- Mode', mode, '--')
    E = simon(key, mode=mode).encrypt
    D = simon(key, mode=mode).decrypt

    encrypted_text = E(my_plaintext)
    print(f"Encrypted text = {encrypted_text}")
    # TODO:: Get binary representation of the ciphertext
    decrypted_text = D(encrypted_text)
    print(f"Decrypted text = {decrypted_text}")


    # for i in range(0,len(msg),16):
    #     pt = msg[i:i+16]
    #     ct = E(bar2int(pt))
    #     pt2 = int2bar(D(ct), 16)
    #     print(' Plaintext in:', pt)
    #     print('   Ciphertext:', hex(ct))
    #     print('Plaintext out:', pt2)