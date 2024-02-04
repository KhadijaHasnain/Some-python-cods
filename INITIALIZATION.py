import numpy as np
from Crypto_functions import *
import time
import sys
import imageio
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(argv):

    arguments = sys.argv[1:]

    pkey = int(arguments[0])
    n_blocks = int(arguments[1])
    enc_times = int(arguments[2])


    s_box_basis_1 = imageio.imread("SBX_114.png")
    s_box_basis_2 = imageio.imread("SBX_112.png")

    enc_key_temp, seed_key_temp = key_generation(pkey)

    enc_key_str = str(enc_key_temp)
    enc_key_ascii = [0]*len(enc_key_str)
    for ce in range(0, len(enc_key_str)):
        enc_key_ascii[ce] = ord(enc_key_str[ce])
    enc_key = np.array(enc_key_ascii)

    seed_key_str = str(seed_key_temp)
    seed_key_ascii = [0]*len(seed_key_str)
    for cs in range(0, len(seed_key_str)):
        seed_key_ascii[cs] = ord(seed_key_str[cs])
    seed_key = np.array(seed_key_ascii)

    s_box1 = s_box_basis_1[int(np.mod(np.sum(seed_key[:int(np.floor(len(seed_key_str)/2))]), 256)),:]
    s_box2 = s_box_basis_2[int(np.mod(np.sum(seed_key[int(np.floor(len(seed_key_str)/2)):]), 256)),:]

    key1, key2, key3, key4 = keys_derivation(enc_key, seed_key_temp, s_box2, m1, m2, m3, m4)

    key_dep_cst = int(np.mod(np.sum(((key1 ^ key2) ^ key3) ^ key4), 33))

    ctr1 = np.empty((n_blocks*enc_times, 16), dtype=int)
    ctr2 = np.empty((n_blocks*enc_times, 16), dtype=int)
    ctr3 = np.empty((n_blocks*enc_times, 16), dtype=int)
    ctr4 = np.empty((n_blocks*enc_times, 16), dtype=int)

    for j in range(enc_times):
        ib1 = chaotic_Hash(key1[j], seed_key_temp, s_box2, 128)
        ib2 = chaotic_Hash(key2[j], seed_key_temp, s_box2, 128)
        ib3 = chaotic_Hash(key3[j], seed_key_temp, s_box2, 128)
        ib4 = chaotic_Hash(key4[j], seed_key_temp, s_box2, 128)
        
        ib1 = chaotic_Rijndael(ib1, key1, s_box2)
        ib2 = chaotic_Rijndael(ib2, key2, s_box2)
        ib3 = chaotic_Rijndael(ib3, key3, s_box2)
        ib4 = chaotic_Rijndael(ib4, key4, s_box2)

        ctr1[n_blocks*j:(n_blocks*j)+(n_blocks)], ctr2[n_blocks*j:(n_blocks*j)+(n_blocks)], \
        ctr3[n_blocks*j:(n_blocks*j)+(n_blocks)], ctr4[n_blocks*j:(n_blocks*j)+(n_blocks)] = \
            counter_compute(n_blocks, ib1[0,:], ib2[0,:], ib3[0,:], ib4[0,:], m1, m2, m3, m4)


    pickle.dump(s_box1, open("s_box_1.p", "wb"))
    pickle.dump(s_box2, open("s_box_2.p", "wb"))
    pickle.dump(key_dep_cst, open("key_dep_cst.p", "wb"))
    pickle.dump(ctr1, open("ctr1.p", "wb"))
    pickle.dump(ctr2, open("ctr2.p", "wb"))
    pickle.dump(ctr3, open("ctr3.p", "wb"))
    pickle.dump(ctr4, open("ctr4.p", "wb"))

    print("--------------------Counters are generated --------------------!")


if __name__ == "__main__":

    main(sys.argv[1:])




