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

    plaintext_type = int(arguments[0])
    encrypt_stages = int(arguments[1])
    plaintext_name = str(arguments[2])

    
    counter_set_1 = pickle.load(open("ctr1.p", "rb"))
    counter_set_2 = pickle.load(open("ctr2.p", "rb"))
    counter_set_3 = pickle.load(open("ctr3.p", "rb"))
    counter_set_4 = pickle.load(open("ctr4.p", "rb"))
    s_box_1 = pickle.load(open("s_box_1.p", "rb"))
    s_box_2 = pickle.load(open("s_box_2.p", "rb"))
    key_dep_cst = pickle.load(open("key_dep_cst.p", "rb"))
    

    msg_init = None
    
    if plaintext_type == 0:
        msg_init = imageio.imread(plaintext_name)
        
    elif plaintext_type == 1:
        with open(plaintext_name, "r") as f:
            msg_temp = f.read()
        ch_ascii = [0] * len(msg_temp)
        for character in range(0, len(msg_temp)):
            ch_ascii[character] = ord(msg_temp[character])
        msg_init = np.array(ch_ascii)

    else:
        ch_ascii = [0]*len(plaintext_name)
        for character in range(0, len(plaintext_name)):
            ch_ascii[character] = ord(plaintext_name[character])
        msg_init = np.array(ch_ascii)




    msg_cube, p, l, dimn, idxx = message_reshaping(msg_init)

    cfr,sudoku_index1, sudoku_index2, s_mode = full_encryption(msg_cube, l, counter_set_1, counter_set_2,
                            counter_set_3,counter_set_4, s_box_1, s_box_2, key_dep_cst, encrypt_stages)

    sudoku_index1 = s_box_2[sudoku_index1]
    sudoku_index2 = s_box_2[sudoku_index2]
    s_mode = s_box_2[s_mode]

    dimn = np.array(dimn, dtype=int).tolist()
    dimn[0] = dimn[0] + idxx
    to_be_saved = cfr.reshape((l*64, 1), order='F')

    pad_data = np.array([p, idxx,  dimn[0]])
    pickle.dump(pad_data, open("pad.p", "wb"))
    
    pickle.dump(np.array(sudoku_index1), open("sudoku_index1.p", "wb"))
    pickle.dump(np.array(sudoku_index2), open("sudoku_index2.p", "wb"))
    pickle.dump(np.array(s_mode), open("s_mode.p", "wb"))

    if plaintext_type == 0:
        to_be_saved = np.reshape(to_be_saved, dimn, order='F')
        imageio.imsave("Encrypted.png",to_be_saved.astype(np.uint8))
        pickle.dump(to_be_saved, open("Encrypted.p", "wb"))

    elif plaintext_type == 1:
        charc = ''
        for i in range(l * 64):
            charc = charc + chr(to_be_saved[i, 0])
        pickle.dump(to_be_saved, open("Encrypted.p", "wb"))
    else:
        charc = ''
        for i in range(l*64):
            charc = charc + chr(to_be_saved[i,0])
        print("-------------------- ciphertext is --------------------: ")
        print(charc)
        pickle.dump(to_be_saved, open("Encrypted.p", "wb"))
        
    
    print("Done")


if __name__ == "__main__":

    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print("it took : ", end - start)




