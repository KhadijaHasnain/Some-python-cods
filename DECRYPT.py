import random
import sys
from Crypto_functions import *
import imageio
import pickle
import time



def main(argv):

    arguments = sys.argv[1:]

    ciphertext_type = int(arguments[0])
    decrypt_stages = int(arguments[1])

    
    counter_set_1 = pickle.load(open("ctr1.p", "rb"))
    counter_set_2 = pickle.load(open("ctr2.p", "rb"))
    counter_set_3 = pickle.load(open("ctr3.p", "rb"))
    counter_set_4 = pickle.load(open("ctr4.p", "rb"))
    s_box_1 = pickle.load(open("s_box_1.p", "rb"))
    s_box_2 = pickle.load(open("s_box_2.p", "rb"))
    sudoku_index1 = pickle.load(open("sudoku_index1.p", "rb"))
    sudoku_index2 = pickle.load(open("sudoku_index2.p", "rb"))
    s_mode = pickle.load(open("s_mode.p", "rb"))


    msg_init = pickle.load(open("Encrypted.p", "rb"))
    pad = pickle.load(open("pad.p", "rb"))

    inv_s_box1 = reverse_idx(s_box_1)[0,:]
    inv_s_box2 = reverse_idx(s_box_2)[0,:]

    sudoku_index1 = inv_s_box2[sudoku_index1]
    sudoku_index2 = inv_s_box2[sudoku_index2]
    s_mode = inv_s_box2[s_mode]

    msg_cube, p, l, dimn, idxx = message_reshaping(msg_init)

    dcr = full_decryption(msg_cube, l, counter_set_1, counter_set_2, counter_set_3, counter_set_4,
                          inv_s_box1, inv_s_box2, sudoku_index1, sudoku_index2, s_mode, decrypt_stages)


    to_be_saved = dcr.reshape((l*64, 1), order='F')
    dimn = np.array(dimn, dtype=int).tolist()
    dimn[0] = pad[2]-pad[1]
    to_be_saved = np.reshape(to_be_saved[:l*64 - pad[0], 0], dimn, order='F')

    if ciphertext_type == 0:
        imageio.imsave("Decrypted.png", to_be_saved.astype(np.uint8))
    elif ciphertext_type == 1:
        charc = ''
        for i in range(l * 64 - pad[0]):
            charc = charc + chr(to_be_saved[i, 0])
        with open('Decrypted.txt', 'w') as f:
            f.write(charc)
            f.close()
    else:
        charc = ''
        for i in range(l * 64 - pad[0]):
            charc = charc + chr(to_be_saved[i, 0])
        print("--------------------Plaintext is --------------------: ")
        print(charc)
        pickle.dump(to_be_saved, open("Decrypted.p", "wb"))

    print(" Done ")


if __name__ == "__main__":
    
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print("it took : ", end - start)

