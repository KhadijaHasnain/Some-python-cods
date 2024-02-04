import numpy as np
from Sudoku_matrix import *
import random


def key_generation(x):
    def modinv(a):
        m = 2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1
        lm, hm = 1, 0
        low, high = a % m, m
        while low > 1:
            ratio = high / low
            nm, new = hm - lm * ratio, high - low * ratio
            lm, low, hm, high = nm, new, lm, low
        return lm % m

    def ec_add(a, b):
        pcurve = 2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1
        lam_add = ((b[1] - a[1]) * modinv(b[0] - a[0])) % pcurve
        x = (lam_add * lam_add - a[0] - b[0]) % pcurve
        y = (lam_add * (a[0] - x) - a[1]) % pcurve
        return x, y

    def ec_double(a):
        pcurve = 2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1
        acurve = 0
        lam = ((3 * a[0] * a[0] + acurve) * modinv((2 * a[1]))) % pcurve
        x = (lam * lam - 2 * a[0]) % pcurve
        y = (lam * (a[0] - x) - a[1]) % pcurve
        return x, y

    scalar_bin = bin(x)[2:]

    bx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
    by = 32670510020758816978083085130507043184471273380659243275938904335757337482424
    base = (bx, by)

    q = base

    for i in range(1, len(scalar_bin)):
        q = ec_double(q)
        if scalar_bin[i] == "1":
            q = ec_add(q, base)

    return int(q[0]), int(q[1])


def keys_derivation(enc_key, seed_key, s_box, m1, m2, m3, m4):

    h1 = chaotic_Hash(enc_key, seed_key, s_box, 128)
    key1 = h1[0, m1[int(np.mod(np.sum(h1[0,:]), 16))]]

    h2 = chaotic_Hash(key1, seed_key, s_box, 128)
    key2 = h2[0, m2[int(np.mod(np.sum(h2[0,:]), 16))]]

    h3 = chaotic_Hash(key2, seed_key, s_box, 128)
    key3 = h3[0, m3[int(np.mod(np.sum(h3[0,:]), 16))]]

    h4 = chaotic_Hash(key3, seed_key, s_box, 128)
    key4 = h4[0, m4[int(np.mod(np.sum(h4[0,:]), 16))]]

    return key1, key2, key3, key4



def circ_right_shift(word, n_bits):
    l = len(word)
    n_bits = n_bits % l
    out = [0] * l
    out[n_bits:l] = word[:l - n_bits]
    out[:n_bits] = word[l - n_bits:]
    out = np.array(out)
    return out


def reverse_idx(arr):
    out = np.empty((1, len(arr)), dtype=int)
    for j in range(len(arr)):
        out[0, j] = int(np.where(np.array(arr) == j)[0])
    return out



def chaotic_Hash(msg, key, s_box, xo):

    def keccak_func(state):

        def bit_not(number):
            b = len(number)
            out = np.empty((1, b), dtype=int)
            for j in range(b):
                out[0, j] = 1 - number[j]
            return out
    
        def bit_and(num1, num2):
            a, b = num1.shape
            out = np.zeros((1, b), dtype=int)
            for j in range(b):
                out[0, j] = num1[0, j] * num2[j]
            return out

        def keccak_scytale(state):
            rod = np.array([2, 3, 4, 6, 7, 8, 12, 14, 16, 21, 24, 28, 32, 42, 48, 56, 64])
            idx = np.mod(np.sum(state), 17)
            state = np.transpose(state.reshape(rod[idx], int(1344/rod[idx]), order='F'))
            out = state.reshape(1, 1344, order='F')
            return out

        def keccak_rubik(state):
        
            def keccak_rubik_90(state):
                out = np.empty((5, 5), dtype=int)

                out[0, 0] = state[0, 4]
                out[1, 0] = state[0, 3]
                out[2, 0] = state[0, 2]
                out[3, 0] = state[0, 1]
                out[4, 0] = state[0, 0]

                out[0, 1] = state[1, 4]
                out[1, 1] = state[1, 3]
                out[2, 1] = state[1, 2]
                out[3, 1] = state[1, 1]
                out[4, 1] = state[1, 0]

                out[0, 2] = state[2, 4]
                out[1, 2] = state[2, 3]
                out[2, 2] = state[2, 2]
                out[3, 2] = state[2, 1]
                out[4, 2] = state[2, 0]

                out[0, 3] = state[3, 4]
                out[1, 3] = state[3, 3]
                out[2, 3] = state[3, 2]
                out[3, 3] = state[3, 1]
                out[4, 3] = state[3, 0]

                out[0, 4] = state[4, 4]
                out[1, 4] = state[4, 3]
                out[2, 4] = state[4, 2]
                out[3, 4] = state[4, 1]
                out[4, 4] = state[4, 0]

                return out

            def keccak_rubik_270(state):
                out = np.empty((5, 5), dtype=int)

                out[0, 0] = state[4, 0]
                out[1, 0] = state[4, 1]
                out[2, 0] = state[4, 2]
                out[3, 0] = state[4, 3]
                out[4, 0] = state[4, 4]

                out[0, 1] = state[3, 0]
                out[1, 1] = state[3, 1]
                out[2, 1] = state[3, 2]
                out[3, 1] = state[3, 3]
                out[4, 1] = state[3, 4]

                out[0, 2] = state[2, 0]
                out[1, 2] = state[2, 1]
                out[2, 2] = state[2, 2]
                out[3, 2] = state[2, 3]
                out[4, 2] = state[2, 4]

                out[0, 3] = state[1, 0]
                out[1, 3] = state[1, 1]
                out[2, 3] = state[1, 2]
                out[3, 3] = state[1, 3]
                out[4, 3] = state[1, 4]

                out[0, 4] = state[0, 0]
                out[1, 4] = state[0, 1]
                out[2, 4] = state[0, 2]
                out[3, 4] = state[0, 3]
                out[4, 4] = state[0, 4]

                return out

            out = np.copy(state)
            x1 = np.array([0, 2, 4])
            y1 = np.array([3, 2])
            z1 = np.array(
                [40, 35, 37, 7, 2, 57, 28, 44, 51, 52, 48, 63, 21, 39, 4, 17, 0, 49, 1, 50, 56, 31, 20, 13, 45, 41, 34, 8,
                 11, 38, 59, 43])


            for i in range(len(x1)):
                out[x1[i], :, :] = np.fliplr(
                    np.transpose(np.fliplr(np.transpose(state[x1[i], :, :]))))
            for j in range(len(y1)):
                out[:, y1[j], :] = np.fliplr(np.transpose(np.fliplr(np.transpose(state[:, y1[j], :]))))
            for k in range(len(z1)):
                out[:, :, z1[k]] = keccak_rubik_90(out[:, :, z1[k]])
            return out


        rc = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        rot = np.array([[25, 39, 3, 10, 43],
           [55, 20, 36, 44, 6],
           [28, 27, 0, 1, 62],
           [56, 14, 18, 2, 61],
           [21, 8, 41, 45, 15]])
        r = 1344
        sr = keccak_scytale(state[:r])
        sc = state[r:]
        state = np.concatenate((sr[0, :], sc))

        k_a = keccak_state_init(state)
        k_c = np.empty((1, 5, 64), dtype=int)
        k_d = np.empty((1, 5, 64), dtype=int)
        k_b = np.empty((5, 5, 64), dtype=int)
        for i in range(10):
            k_c[0, 2, :] = (((k_a[2, 2, :] ^ k_a[1, 2, :]) ^ k_a[0, 2, :]) ^ k_a[4, 2, :]) ^ k_a[3, 2, :]
            k_c[0, 3, :] = (((k_a[2, 3, :] ^ k_a[1, 3, :]) ^ k_a[0, 3, :]) ^ k_a[4, 3, :]) ^ k_a[3, 3, :]
            k_c[0, 4, :] = (((k_a[2, 4, :] ^ k_a[1, 4, :]) ^ k_a[0, 4, :]) ^ k_a[4, 4, :]) ^ k_a[3, 4, :]
            k_c[0, 0, :] = (((k_a[2, 0, :] ^ k_a[1, 0, :]) ^ k_a[0, 0, :]) ^ k_a[4, 0, :]) ^ k_a[3, 0, :]
            k_c[0, 1, :] = (((k_a[2, 1, :] ^ k_a[1, 1, :]) ^ k_a[0, 1, :]) ^ k_a[4, 1, :]) ^ k_a[3, 1, :]

            k_d[0, 2, :] = k_c[0, 1, :] ^ circ_right_shift(k_c[0, 3, :], 1)
            k_d[0, 3, :] = k_c[0, 2, :] ^ circ_right_shift(k_c[0, 4, :], 1)
            k_d[0, 4, :] = k_c[0, 3, :] ^ circ_right_shift(k_c[0, 0, :], 1)
            k_d[0, 0, :] = k_c[0, 4, :] ^ circ_right_shift(k_c[0, 1, :], 1)
            k_d[0, 1, :] = k_c[0, 0, :] ^ circ_right_shift(k_c[0, 2, :], 1)

            k_a[2, 2, :] = k_a[2, 2, :] ^ k_d[0, 2, :]
            k_a[1, 2, :] = k_a[1, 2, :] ^ k_d[0, 2, :]
            k_a[0, 2, :] = k_a[0, 2, :] ^ k_d[0, 2, :]
            k_a[4, 2, :] = k_a[4, 2, :] ^ k_d[0, 2, :]
            k_a[3, 2, :] = k_a[3, 2, :] ^ k_d[0, 2, :]
            k_a[2, 3, :] = k_a[2, 3, :] ^ k_d[0, 3, :]
            k_a[1, 3, :] = k_a[1, 3, :] ^ k_d[0, 3, :]
            k_a[0, 3, :] = k_a[0, 3, :] ^ k_d[0, 3, :]
            k_a[4, 3, :] = k_a[4, 3, :] ^ k_d[0, 3, :]
            k_a[3, 3, :] = k_a[3, 3, :] ^ k_d[0, 3, :]
            k_a[2, 4, :] = k_a[2, 4, :] ^ k_d[0, 4, :]
            k_a[1, 4, :] = k_a[1, 4, :] ^ k_d[0, 4, :]
            k_a[0, 4, :] = k_a[0, 4, :] ^ k_d[0, 4, :]
            k_a[4, 4, :] = k_a[4, 4, :] ^ k_d[0, 4, :]
            k_a[3, 4, :] = k_a[3, 4, :] ^ k_d[0, 4, :]
            k_a[2, 0, :] = k_a[2, 0, :] ^ k_d[0, 0, :]
            k_a[1, 0, :] = k_a[1, 0, :] ^ k_d[0, 0, :]
            k_a[0, 0, :] = k_a[0, 0, :] ^ k_d[0, 0, :]
            k_a[4, 0, :] = k_a[4, 0, :] ^ k_d[0, 0, :]
            k_a[3, 0, :] = k_a[3, 0, :] ^ k_d[0, 0, :]
            k_a[2, 1, :] = k_a[2, 1, :] ^ k_d[0, 1, :]
            k_a[1, 1, :] = k_a[1, 1, :] ^ k_d[0, 1, :]
            k_a[0, 1, :] = k_a[0, 1, :] ^ k_d[0, 1, :]
            k_a[4, 1, :] = k_a[4, 1, :] ^ k_d[0, 1, :]
            k_a[3, 1, :] = k_a[3, 1, :] ^ k_d[0, 1, :]

            k_b[2, 2, :] = circ_right_shift(k_a[2, 2, :], rot[2, 2])
            k_b[4, 3, :] = circ_right_shift(k_a[1, 2, :], rot[1, 2])
            k_b[1, 4, :] = circ_right_shift(k_a[0, 2, :], rot[0, 2])
            k_b[3, 0, :] = circ_right_shift(k_a[4, 2, :], rot[4, 2])
            k_b[0, 1, :] = circ_right_shift(k_a[3, 2, :], rot[3, 2])
            k_b[0, 2, :] = circ_right_shift(k_a[2, 3, :], rot[2, 3])
            k_b[2, 3, :] = circ_right_shift(k_a[1, 3, :], rot[1, 3])
            k_b[4, 4, :] = circ_right_shift(k_a[0, 3, :], rot[0, 3])
            k_b[1, 0, :] = circ_right_shift(k_a[4, 3, :], rot[4, 3])
            k_b[3, 1, :] = circ_right_shift(k_a[3, 3, :], rot[3, 3])
            k_b[3, 2, :] = circ_right_shift(k_a[2, 4, :], rot[2, 4])
            k_b[0, 3, :] = circ_right_shift(k_a[1, 4, :], rot[1, 4])
            k_b[2, 4, :] = circ_right_shift(k_a[0, 4, :], rot[0, 4])
            k_b[4, 0, :] = circ_right_shift(k_a[4, 4, :], rot[4, 4])
            k_b[1, 1, :] = circ_right_shift(k_a[3, 4, :], rot[3, 4])
            k_b[1, 2, :] = circ_right_shift(k_a[2, 0, :], rot[2, 0])
            k_b[3, 3, :] = circ_right_shift(k_a[1, 0, :], rot[1, 0])
            k_b[0, 4, :] = circ_right_shift(k_a[0, 0, :], rot[0, 0])
            k_b[2, 0, :] = circ_right_shift(k_a[4, 0, :], rot[4, 0])
            k_b[4, 1, :] = circ_right_shift(k_a[3, 0, :], rot[3, 0])
            k_b[4, 2, :] = circ_right_shift(k_a[2, 1, :], rot[2, 1])
            k_b[1, 3, :] = circ_right_shift(k_a[1, 1, :], rot[1, 1])
            k_b[3, 4, :] = circ_right_shift(k_a[0, 1, :], rot[0, 1])
            k_b[0, 0, :] = circ_right_shift(k_a[4, 1, :], rot[4, 1])
            k_b[2, 1, :] = circ_right_shift(k_a[3, 1, :], rot[3, 1])

            k_a[2, 2, :] = k_b[2, 2, :] ^ bit_and(bit_not(k_b[2, 3, :]), k_b[2, 4, :])
            k_a[1, 2, :] = k_b[1, 2, :] ^ bit_and(bit_not(k_b[1, 3, :]), k_b[1, 4, :])
            k_a[0, 2, :] = k_b[0, 2, :] ^ bit_and(bit_not(k_b[0, 3, :]), k_b[0, 4, :])
            k_a[4, 2, :] = k_b[4, 2, :] ^ bit_and(bit_not(k_b[4, 3, :]), k_b[4, 4, :])
            k_a[3, 2, :] = k_b[3, 2, :] ^ bit_and(bit_not(k_b[3, 3, :]), k_b[3, 4, :])
            k_a[2, 3, :] = k_b[2, 3, :] ^ bit_and(bit_not(k_b[2, 3, :]), k_b[2, 4, :])
            k_a[1, 3, :] = k_b[1, 3, :] ^ bit_and(bit_not(k_b[1, 3, :]), k_b[1, 4, :])
            k_a[0, 3, :] = k_b[0, 3, :] ^ bit_and(bit_not(k_b[0, 3, :]), k_b[0, 4, :])
            k_a[4, 3, :] = k_b[4, 3, :] ^ bit_and(bit_not(k_b[4, 3, :]), k_b[4, 4, :])
            k_a[3, 3, :] = k_b[3, 3, :] ^ bit_and(bit_not(k_b[3, 3, :]), k_b[3, 4, :])
            k_a[2, 4, :] = k_b[2, 4, :] ^ bit_and(bit_not(k_b[2, 3, :]), k_b[2, 4, :])
            k_a[1, 4, :] = k_b[1, 4, :] ^ bit_and(bit_not(k_b[1, 3, :]), k_b[1, 4, :])
            k_a[0, 4, :] = k_b[0, 4, :] ^ bit_and(bit_not(k_b[0, 3, :]), k_b[0, 4, :])
            k_a[4, 4, :] = k_b[4, 4, :] ^ bit_and(bit_not(k_b[4, 3, :]), k_b[4, 4, :])
            k_a[3, 4, :] = k_b[3, 4, :] ^ bit_and(bit_not(k_b[3, 3, :]), k_b[3, 4, :])
            k_a[2, 0, :] = k_b[2, 0, :] ^ bit_and(bit_not(k_b[2, 3, :]), k_b[2, 4, :])
            k_a[1, 0, :] = k_b[1, 0, :] ^ bit_and(bit_not(k_b[1, 3, :]), k_b[1, 4, :])
            k_a[0, 0, :] = k_b[0, 0, :] ^ bit_and(bit_not(k_b[0, 3, :]), k_b[0, 4, :])
            k_a[4, 0, :] = k_b[4, 0, :] ^ bit_and(bit_not(k_b[4, 3, :]), k_b[4, 4, :])
            k_a[3, 0, :] = k_b[3, 0, :] ^ bit_and(bit_not(k_b[3, 3, :]), k_b[3, 4, :])
            k_a[2, 1, :] = k_b[2, 1, :] ^ bit_and(bit_not(k_b[2, 3, :]), k_b[2, 4, :])
            k_a[1, 1, :] = k_b[1, 1, :] ^ bit_and(bit_not(k_b[1, 3, :]), k_b[1, 4, :])
            k_a[0, 1, :] = k_b[0, 1, :] ^ bit_and(bit_not(k_b[0, 3, :]), k_b[0, 4, :])
            k_a[4, 1, :] = k_b[4, 1, :] ^ bit_and(bit_not(k_b[4, 3, :]), k_b[4, 4, :])
            k_a[3, 1, :] = k_b[3, 1, :] ^ bit_and(bit_not(k_b[3, 3, :]), k_b[3, 4, :])

            k_a[2, 2, :] = k_a[2, 2, :] ^ rc[i, :]
            
        k_a = keccak_rubik(k_a)

        return k_a

    def keccak_state_init(state):
        words = np.empty((25, 64), dtype=int)
        out = np.empty((5, 5, 64), dtype=int)
        for i in range(25):
            words[i, :] = state[64*i:(64*i)+64]
        out[2, 2, :] = words[0, :]
        out[1, 2, :] = words[1, :]
        out[0, 2, :] = words[2, :]
        out[4, 2, :] = words[3, :]
        out[3, 2, :] = words[4, :]

        out[2, 3, :] = words[5, :]
        out[1, 3, :] = words[6, :]
        out[0, 3, :] = words[7, :]
        out[4, 3, :] = words[8, :]
        out[3, 3, :] = words[9, :]

        out[2, 4, :] = words[10, :]
        out[1, 4, :] = words[11, :]
        out[0, 4, :] = words[12, :]
        out[4, 4, :] = words[13, :]
        out[3, 4, :] = words[14, :]

        out[2, 0, :] = words[15, :]
        out[1, 0, :] = words[16, :]
        out[0, 0, :] = words[17, :]
        out[4, 0, :] = words[18, :]
        out[3, 0, :] = words[19, :]

        out[2, 1, :] = words[20, :]
        out[1, 1, :] = words[21, :]
        out[0, 1, :] = words[22, :]
        out[4, 1, :] = words[23, :]
        out[3, 1, :] = words[24, :]
        
        return out

    def inv_keccak_state(state):
        out = np.empty((25, 64), dtype=int)
        out[0, :] = state[2, 2, :]
        out[1, :] = state[1, 2, :]
        out[2, :] = state[0, 2, :]
        out[3, :] = state[4, 2, :]
        out[4, :] = state[3, 2, :]

        out[5, :] = state[2, 3, :]
        out[6, :] = state[1, 3, :]
        out[7, :] = state[0, 3, :]
        out[8, :] = state[4, 3, :]
        out[9, :] = state[3, 3, :]

        out[10, :] = state[2, 4, :]
        out[11, :] = state[1, 4, :]
        out[12, :] = state[0, 4, :]
        out[13, :] = state[4, 4, :]
        out[14, :] = state[3, 4, :]

        out[15, :] = state[2, 0, :]
        out[16, :] = state[1, 0, :]
        out[17, :] = state[0, 0, :]
        out[18, :] = state[4, 0, :]
        out[19, :] = state[3, 0, :]

        out[20, :] = state[2, 1, :]
        out[21, :] = state[1, 1, :]
        out[22, :] = state[0, 1, :]
        out[23, :] = state[4, 1, :]
        out[24, :] = state[3, 1, :]

        out = out.reshape(1, 1600, order='F')
        
        return out


    def hash_algorithm(msg, xo):
        r = 1344
        c = 256
        sr = np.zeros((1, r), dtype=int)
        sc = np.zeros((1, c), dtype=int)
        if np.mod(len(msg), r) > 0:
            pad = np.zeros((1, int((r - np.mod(len(msg), r)))), dtype=int)
            if len(pad[0,:]) >= 6:
                pad[0, :5] = 1
                pad[0, len(pad[0, :])-1] = 1
            else:
                pad[0,:] = 1
                
            msg_pad = np.concatenate((msg, pad[0, :]))
        else:
            msg_pad = msg
        state = None
        for i in range(int(len(msg_pad)/r)):
            blk = msg_pad[r*i: (r*i)+r]
            sp = sr ^ blk
            state = np.concatenate((sp[0, :], sc[0, :]))
            state = keccak_func(state)
            state = inv_keccak_state(state)
            sr = state[0, :r]
            sc = state[0, r+1:]
        sr_temp = state[0, :r]

        out = sr_temp[:xo]

        return out

    def right_encode(x):
        if x == 0:
            ss = np.array([0])
        else:
            s = np.empty((1, 1), dtype=int)
            while x > 0:
                i = 0
                s[i] = np.mod(x, 256)
                x = np.floor(x / 256)
                i = i + 1
            ss = np.array([np.fliplr(s), len(s)], dtype=object)
        a = np.prod(ss.shape)
        bin_ss = ((ss.reshape((-1, 1), order='F') & (2 ** np.arange(8))) != 0).astype(np.uint8)
        bin_ss = bin_ss.reshape((1, 8 * a))
        return bin_ss

    msg = np.mod(msg, 256)
    lm = int(np.prod(msg.shape))
    msg = np.reshape(msg, (lm, 1), order='F')
    for i in range(lm):
        msg[i] = s_box[int(msg[i])]
    msg = np.array(msg)

    bin_msg = ((msg.reshape((-1, 1), order='F') & (2 ** np.arange(8))) != 0).astype(np.uint8)
    bin_msg = bin_msg.reshape((1, 8 * lm))
    bin_key = np.binary_repr(key, width=8)[::-1]
    bin_key = " ".join(bin_key)
    bin_key = bin_key.split()
    bin_key = [int(j) for j in bin_key]
    bin_key = np.array([bin_key])
    rek = right_encode(len(bin_key))

    s = np.concatenate((bin_msg[0, :], bin_key[0, :], rek[0, :]))

    h = hash_algorithm(np.concatenate((s, np.array([1, 1]))), xo)
    res = h.astype(str)
    res = [''.join(idx for idx in sub) for sub in res]
    res = ''.join(res)

    out = np.empty((1, int(xo/8)), dtype=int)
    for i in range(int(xo/8)):
        out[0,i] = int(res[8*i:8*i+8], 2)

    return out





def chaotic_Rijndael(state, key, s_box):

    def mix_col(state,r_mode):

        if r_mode == 1:
            state = block_rubik_90(state)

        mds = np.array([[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]])
        out = np.empty((4, 4), dtype=int)

        out[0,0] = (((mds[0, 0] * state[0, 0]) ^ (mds[0, 1] * state[1, 0])) ^ (mds[0, 2] * state[2, 0])) ^ (
                mds[0, 3] * state[3, 0])
        out[1,0] = (((mds[1, 0] * state[0, 0]) ^ (mds[1, 1] * state[1, 0])) ^ (mds[1, 2] * state[2, 0])) ^ (
                mds[1, 3] * state[3, 0])
        out[2,0] = (((mds[2, 0] * state[0, 0]) ^ (mds[2, 1] * state[1, 0])) ^ (mds[2, 2] * state[2, 0])) ^ (
                mds[2, 3] * state[3, 0])
        out[3,0] = (((mds[3, 0] * state[0, 0]) ^ (mds[3, 1] * state[1, 0])) ^ (mds[3, 2] * state[2, 0])) ^ (
                mds[3, 3] * state[3, 0])

        out[0,1] = (((mds[0, 0] * state[0, 1]) ^ (mds[0, 1] * state[1, 1])) ^ (mds[0, 2] * state[2, 1])) ^ (
                mds[0, 3] * state[3, 1])
        out[1,1] = (((mds[1, 0] * state[0, 1]) ^ (mds[1, 1] * state[1, 1])) ^ (mds[1, 2] * state[2, 1])) ^ (
                mds[1, 3] * state[3, 1])
        out[2,1] = (((mds[2, 0] * state[0, 1]) ^ (mds[2, 1] * state[1, 1])) ^ (mds[2, 2] * state[2, 1])) ^ (
                mds[2, 3] * state[3, 1])
        out[3,1] = (((mds[3, 0] * state[0, 1]) ^ (mds[3, 1] * state[1, 1])) ^ (mds[3, 2] * state[2, 1])) ^ (
                mds[3, 3] * state[3, 1])

        out[0,2] = (((mds[0, 0] * state[0, 2]) ^ (mds[0, 1] * state[1, 2])) ^ (mds[0, 2] * state[2, 2])) ^ (
                mds[0, 3] * state[3, 2])
        out[1,2] = (((mds[1, 0] * state[0, 2]) ^ (mds[1, 1] * state[1, 2])) ^ (mds[1, 2] * state[2, 2])) ^ (
                mds[1, 3] * state[3, 2])
        out[2,2] = (((mds[2, 0] * state[0, 2]) ^ (mds[2, 1] * state[1, 2])) ^ (mds[2, 2] * state[2, 2])) ^ (
                mds[2, 3] * state[3, 0])
        out[3,2] = (((mds[3, 0] * state[0, 2]) ^ (mds[3, 1] * state[1, 2])) ^ (mds[3, 2] * state[2, 2])) ^ (
                mds[3, 3] * state[3, 2])

        out[0,3] = (((mds[0, 0] * state[0, 3]) ^ (mds[0, 1] * state[1, 3])) ^ (mds[0, 2] * state[2, 3])) ^ (
                mds[0, 3] * state[3, 3])
        out[1,3] = (((mds[1, 0] * state[0, 3]) ^ (mds[1, 1] * state[1, 3])) ^ (mds[1, 2] * state[2, 3])) ^ (
                mds[1, 3] * state[3, 3])
        out[2,3] = (((mds[2, 0] * state[0, 3]) ^ (mds[2, 1] * state[1, 3])) ^ (mds[2, 2] * state[2, 3])) ^ (
                mds[2, 3] * state[3, 0])
        out[3,3] = (((mds[3, 0] * state[0, 3]) ^ (mds[3, 1] * state[1, 3])) ^ (mds[3, 2] * state[2, 3])) ^ (
                mds[3, 3] * state[3, 3])
        return out

    def shift_row(state):
        out = np.empty((4, 4), dtype=int)

        out[0, 0] = state[0, 0]
        out[0, 1] = state[0, 1]
        out[0, 2] = state[0, 2]
        out[0, 3] = state[0, 3]

        out[1, 0] = state[1, 1]
        out[1, 1] = state[1, 2]
        out[1, 2] = state[1, 3]
        out[1, 3] = state[1, 0]

        out[2, 0] = state[2, 2]
        out[2, 1] = state[2, 3]
        out[2, 2] = state[2, 0]
        out[2, 3] = state[2, 1]

        out[3, 0] = state[3, 3]
        out[3, 1] = state[3, 0]
        out[3, 2] = state[3, 1]
        out[3, 3] = state[3, 2]

        return out

    def key_expansion(key, s_box):
        def wrap(x):
            y = np.empty((1, len(x)), dtype=int)
            for j in range(len(x)):
                y[0, j] = np.mod(x[j], 16)
            return y
        
        key = key.reshape((4, 4), order='F')
        s_box = np.reshape(s_box, (16, 16), order='F')
        keyexp = np.empty((32, 4), dtype=int)
        keyexp[0, :] = key[0, :]
        keyexp[1, :] = key[1, :]
        keyexp[2, :] = key[2, :]
        keyexp[3, :] = key[3, :]
        i1, j1 = np.where(s_box == key[0, 0])
        i2, j2 = np.where(s_box == key[0, 1])
        i3, j3 = np.where(s_box == key[0, 2])
        i4, j4 = np.where(s_box == key[0, 3])
        i5, j5 = np.where(s_box == key[1, 0])
        i6, j6 = np.where(s_box == key[1, 1])
        i7, j7 = np.where(s_box == key[1, 2])
        i8, j8 = np.where(s_box == key[1, 3])
        i9, j9 = np.where(s_box == key[2, 0])
        i10, j10 = np.where(s_box == key[2, 1])
        i11, j11 = np.where(s_box == key[2, 2])
        i12, j12 = np.where(s_box == key[2, 3])
        i13, j13 = np.where(s_box == key[3, 0])
        i14, j14 = np.where(s_box == key[3, 1])
        i15, j15 = np.where(s_box == key[3, 2])
        i16, j16 = np.where(s_box == key[3, 3])

        keyexp[4, :] = np.array([key[0, 0], key[0, 0], key[0, 0], key[0, 0]]) ^ s_box[i1, wrap(np.array([j1+1, j1+2, j1+3, j1+4]))]
        keyexp[5, :] = np.array([key[0, 1], key[0, 1], key[0, 1], key[0, 1]]) ^ s_box[i2, wrap(np.array([j2+1, j2+2, j2+3, j2+4]))]
        keyexp[6, :] = np.array([key[0, 2], key[0, 2], key[0, 2], key[0, 2]]) ^ s_box[i3, wrap(np.array([j3+1, j3+2, j3+3, j3+4]))]
        keyexp[7, :] = np.array([key[0, 3], key[0, 3], key[0, 3], key[0, 3]]) ^ s_box[i4, wrap(np.array([j4+1, j4+2, j4+3, j4+4]))]

        keyexp[8, :] = np.array([key[1, 0], key[1, 0], key[1, 0], key[1, 0]]) ^ s_box[wrap(np.array([i5+1, i5+2, i5+3, i5+4])), j5]
        keyexp[9, :] = np.array([key[1, 1], key[1, 1], key[1, 1], key[1, 1]]) ^ s_box[wrap(np.array([i6+1, i6+2, i6+3, i6+4])), j6]
        keyexp[10, :] = np.array([key[1, 2], key[1, 2], key[1, 2], key[1, 2]]) ^ s_box[wrap(np.array([i7+1, i7+2, i7+3, i7+4])), j7]
        keyexp[11, :] = np.array([key[1, 3], key[1, 3], key[1, 3], key[1, 3]]) ^ s_box[wrap(np.array([i14+1, i14+2, i14+3, i14+4])), j14]

        keyexp[12, :] = np.array([key[2, 0], key[2, 0], key[2, 0], key[2, 0]]) ^ s_box[i9, wrap(np.array([j9-4, j9-3, j9-2, j9-1]))]
        keyexp[13, :] = np.array([key[2, 1], key[2, 1], key[2, 1], key[2, 1]]) ^ s_box[i10, wrap(np.array([j10-4, j10-3, j10-2, j10-1]))]
        keyexp[14, :] = np.array([key[2, 2], key[2, 2], key[2, 2], key[2, 2]]) ^ s_box[i11, wrap(np.array([j11-4, j11-3, j11-2, j11-1]))]
        keyexp[15, :] = np.array([key[2, 3], key[2, 3], key[2, 3], key[2, 3]]) ^ s_box[i12, wrap(np.array([j12-4, j12-3, j12-2, j12-1]))]

        keyexp[16, :] = np.array([key[3, 0], key[3, 0], key[3, 0], key[3, 0]]) ^ s_box[wrap(np.array([i4-4, i13-3, i13-2, i13-1])), j13]
        keyexp[17, :] = np.array([key[3, 1], key[3, 1], key[3, 1], key[3, 1]]) ^ s_box[wrap(np.array([i14-4, i14-3, i14-2, i14-1])), j14]
        keyexp[18, :] = np.array([key[3, 2], key[3, 2], key[3, 2], key[3, 2]]) ^ s_box[wrap(np.array([i15-4, i15-3, i15-2, i15-1])), j15]
        keyexp[19, :] = np.array([key[3, 3], key[3, 3], key[3, 3], key[3, 3]]) ^ s_box[wrap(np.array([i16-4, i16-3, i16-2, i16-1])), j16]

        p1 = np.array([key[0, 0], key[0, 0]]) ^ s_box[i1, wrap(np.array([j1 + 1, j1 + 2]))]
        p2 = np.array([key[3, 3], key[3, 3]]) ^ s_box[i16, wrap(np.array([j16 + 1, j16 + 2]))]
        keyexp[20, :] = np.concatenate((p1[0, :], p2[0, :]))
        p1 = np.array([key[2, 0], key[2, 0]]) ^ s_box[wrap(np.array([i9+1, i9+2])), j9]
        p2 = np.array([key[1, 3], key[1, 3]]) ^ s_box[wrap(np.array([i8+1, i8+2])), j8]
        keyexp[21, :] = np.concatenate((p1[0, :], p2[0, :]))
        p1 = np.array([key[0, 1], key[0, 1]]) ^ s_box[i2, wrap(np.array([j2-2, j2-1]))]
        p2 = np.array([key[3, 2], key[3, 2]]) ^ s_box[i15, wrap(np.array([j15-2, j15-1]))]
        keyexp[22, :] = np.concatenate((p1[0, :], p2[0, :]))
        p1 = np.array([key[2, 1], key[2, 1]]) ^ s_box[wrap(np.array([i10-2, i10-1])), j10]
        p2 = np.array([key[1, 2], key[1, 2]]) ^ s_box[wrap(np.array([i7-2, i7-1])), j7]
        keyexp[23, :] = np.concatenate((p1[0, :], p2[0, :]))

        p1 = np.array([key[3, 1], key[3, 1]]) ^ s_box[wrap(np.array([i14+1, i14+2])), j14]
        p2 = np.array([key[0, 2], key[0, 2]]) ^ s_box[wrap(np.array([i3+1, i3+2])), j3]
        keyexp[24, :] = np.concatenate((p1[0, :], p2[0, :]))
        p1 = np.array([key[1, 1], key[1, 1]]) ^ s_box[i6, wrap(np.array([j6+1, j6+2]))]
        p2 = np.array([key[2, 2], key[2, 2]]) ^ s_box[i11, wrap(np.array([j11+1, j11+2]))]
        keyexp[25, :] = np.concatenate((p1[0, :], p2[0, :]))
        p1 = np.array([key[3, 0], key[3, 0]]) ^ s_box[wrap(np.array([i13-2, i13-1])), j13]
        p2 = np.array([key[0, 3], key[0, 3]]) ^ s_box[wrap(np.array([i4-2, i4-1])), j4]
        keyexp[26, :] = np.concatenate((p1[0, :], p2[0, :]))
        p1 = np.array([key[1, 0], key[1, 0]]) ^ s_box[i5, wrap(np.array([j5-2, j5-1]))]
        p2 = np.array([key[2, 3], key[2, 3]]) ^ s_box[i12, wrap(np.array([j12-2, j12-1]))]
        keyexp[27, :] = np.concatenate((p1[0, :], p2[0, :]))

        p1 = key[1, 0] ^ s_box[i5, wrap(j5+1)]
        p2 = key[2, 0] ^ s_box[wrap(i9+1), j9]
        p3 = key[3, 0] ^ s_box[i13, wrap(j13-1)]
        p4 = key[0, 0] ^ s_box[wrap(i1-1), j1]
        keyexp[28, :] = np.concatenate((p1[0, :], p2[0, :], p3[0, :], p4[0, :]))
        p1 = key[1, 1] ^ s_box[wrap(i6+1), j6]
        p2 = key[2, 1] ^ s_box[i10, wrap(j10-1)]
        p3 = key[3, 1] ^ s_box[wrap(i14-1), j14]
        p4 = key[0, 1] ^ s_box[i2, wrap(j2+1)]
        keyexp[29, :] = np.concatenate((p1[0, :], p2[0, :], p3[0, :], p4[0, :]))
        p1 = key[1, 2] ^ s_box[i7, wrap(j7+1)]
        p2 = key[2, 2] ^ s_box[wrap(i1+1), j11]
        p3 = key[3, 2] ^ s_box[i15, wrap(j15-1)]
        p4 = key[0, 2] ^ s_box[wrap(i3-1), j3]
        keyexp[30, :] = np.concatenate((p1[0, :], p2[0, :], p3[0, :], p4[0, :]))
        p1 = key[1, 3] ^ s_box[i8, wrap(j8+1)]
        p2 = key[2, 3] ^ s_box[i12, wrap(j12-1)]
        p3 = key[3, 3] ^ s_box[wrap(i16-1), i16]
        p4 = key[0, 3] ^ s_box[i4, wrap(j4+1)]
        keyexp[31, :] = np.concatenate((p1[0, :], p2[0, :], p3[0, :], p4[0, :]))

        keyexp = np.mod(keyexp, 256)
    
        return keyexp


    def rijndael_rubik(state):

        def rijndael_rubik_90(state):
            out = np.empty((2, 2), dtype=int)
            out[0, 0] = state[1, 0]
            out[0, 1] = state[0, 0]
            out[1, 0] = state[1, 1]
            out[1, 1] = state[1, 0]
            return out

        def rijndael_rubik_270(state):
            out = np.empty((2, 2), dtype=int)
            out[0, 0] = state[0, 1]
            out[0, 1] = state[1, 1]
            out[1, 0] = state[0, 0]
            out[1, 1] = state[1, 0]
            return out

        state = state.reshape((2, 2, 4))
        out = np.copy(state)
        x = np.array([0])
        y = np.array([1, 0])
        z1 = np.array([0, 2, 1])
        z2 = np.array([1, 3])
        for i in range(len(x)):
            out[x[i], :, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(state[x[i], :, :]))))
        for j in range(len(y)):
            out[:, y[j], :] = np.fliplr(np.transpose(np.fliplr(np.transpose(state[:, y[j], :]))))
        for k in range(len(z1)):
            out[:, :, z1[k]] = rijndael_rubik_90(out[:, :, z1[k]])
        for q in range(len(z2)):
            out[:, :, z2[q]] = rijndael_rubik_270(out[:, :, z2[q]])

        return out


    state = state.reshape((4, 4), order='F')
    round_keys = key_expansion(key, s_box)

    state = np.mod(state ^ round_keys[0:4, :], 256)

    for i in range(7):
        state = np.mod(state ^ round_keys[(4 * (i+1)):(4 * (i+1)) + 4, :], 256)
        state_temp = state.flatten().tolist()
        for i in range(16):
            state_temp[i] = s_box[state_temp[i]]
        state = np.reshape(state_temp, (4, 4), order='F')
        state = shift_row(state)
        state = np.mod(mix_col(state, 0), 256)
        state = np.mod(mix_col(state, 1), 256)

    state_temp = state.flatten().tolist()
    for i in range(16):
        state_temp[i] = s_box[state_temp[i]]
    state = np.reshape(state_temp, (4, 4))
    key_mixed = np.array([[round_keys[0, 0], round_keys[7, 1], round_keys[10, 2], round_keys[13, 3]],
                        [round_keys[17, 0], round_keys[20, 1], round_keys[27, 2], round_keys[30, 3]],
                        [round_keys[2, 0], round_keys[5, 1], round_keys[8, 2], round_keys[15, 3]],
                        [round_keys[19, 0], round_keys[22, 1], round_keys[25, 2], round_keys[28, 3]]])
    state = np.mod(state ^ key_mixed, 256)

    state = rijndael_rubik(state)

    state = state.reshape((1, 16), order='F')
    return state


def counter_compute(l, ib1, ib2, ib3, ib4, m1, m2, m3, m4):
    # l is the number of 64-bytes blocks (4x4x4 cubes) included in the message
    
    mtrix = {0: m1, 1: m2, 2: m3, 3: m4}
    n = 16
    m = 4
    s1 = np.sum(ib1)
    s2 = np.sum(ib2)
    s3 = np.sum(ib3)
    s4 = np.sum(ib4)

    ctr1 = np.empty((l, 16), dtype=int)
    ctr2 = np.empty((l, 16), dtype=int)
    ctr3 = np.empty((l, 16), dtype=int)
    ctr4 = np.empty((l, 16), dtype=int)

    for i in range(l):
        i_bin = np.binary_repr(i, width=8)[::-1]
        n_bytes = int(np.floor(len(i_bin) / 8))
        ctr_temp = [0] * (n_bytes+1)

        for j in range(n_bytes):
            ctr_temp[j] = int(i_bin[8 * j:(8 * j) + 8], 2)
        ctr_temp[n_bytes] = int(i_bin[(8 * j):], 2)

        ib1 = ib1[mtrix[np.mod(s1, m)][np.mod(i, n)]]
        ib2 = ib2[mtrix[np.mod(s2, m)][np.mod(i, n)]]
        ib3 = ib3[mtrix[np.mod(s3, m)][np.mod(i, n)]]
        ib4 = ib4[mtrix[np.mod(s4, m)][np.mod(i, n)]]

        ctr1[i, :] = ib1
        ctr2[i, :] = ib2
        ctr3[i, :] = ib3
        ctr4[i, :] = ib4

        ctr1[i, :] = np.concatenate((ctr1[i, :16-len(ctr_temp)],  np.array(ctr_temp)))
        ctr2[i, :] = np.concatenate((ctr2[i, :16 - len(ctr_temp)], np.array(ctr_temp)))
        ctr3[i, :] = np.concatenate((ctr3[i, :16 - len(ctr_temp)], np.array(ctr_temp)))
        ctr4[i, :] = np.concatenate((ctr4[i, :16 - len(ctr_temp)], np.array(ctr_temp)))

        s1 = np.sum(ctr1[i, :])
        s2 = np.sum(ctr2[i, :])
        s3 = np.sum(ctr3[i, :])
        s4 = np.sum(ctr4[i, :])

    return np.mod(ctr1, 256), np.mod(ctr2, 256), np.mod(ctr3, 256), np.mod(ctr4, 256)



def bloc_encrypt(msg, ctr1, ctr2, ctr3, ctr4, s_mode):
    
    def cipher_round(a,b,c,d,ctr1,ctr2,ctr3,ctr4):
        aa = np.reshape(a, (1, 16), order='F')
        bb = np.reshape(b, (1, 16), order='F')
        cc = np.reshape(c, (1, 16), order='F')
        dd = np.reshape(d, (1, 16), order='F')
        aa = aa ^ ctr1
        dd = aa ^ dd
        bb = aa ^ bb
        bb = bb ^ ctr2
        bb = bb ^ cc
        cc = cc ^ ctr3
        cc = cc ^ bb
        dd = dd ^ cc
        dd = dd ^ ctr4
        aa = aa ^ dd
        aa = np.mod(np.reshape(aa, (4, 4), order='F'), 256)
        bb = np.mod(np.reshape(bb, (4, 4), order='F'), 256)
        cc = np.mod(np.reshape(cc, (4, 4), order='F'), 256)
        dd = np.mod(np.reshape(dd, (4, 4), order='F'), 256)
        return aa, bb, cc, dd
    
    x, y, z = msg.shape
    n = int(z/4)
    out = np.empty((x, y, z), dtype=np.uint8)

 
    if s_mode == 1:
        for i in range(n):
            a = msg[0, :, 4*i:(4*i)+4]
            b = msg[1, :, 4*i:(4*i)+4]
            c = msg[2, :, 4*i:(4*i)+4]
            d = msg[3, :, 4*i:(4*i)+4]
            aa, bb, cc, dd = cipher_round(a, b, c, d, ctr1[i, :], ctr2[i, :], ctr3[i, :], ctr4[i, :])
            out[0, :, 4*i:(4*i)+4] = aa
            out[1, :, 4*i:(4*i)+4] = bb
            out[2, :, 4*i:(4*i)+4] = cc
            out[3, :, 4*i:(4*i)+4] = dd
    elif s_mode == 2:
        for i in range(n):
            a = msg[:, 0, 4*i:(4*i)+4]
            b = msg[:, 1, 4*i:(4*i)+4]
            c = msg[:, 2, 4*i:(4*i)+4]
            d = msg[:, 3, 4*i:(4*i)+4]
            aa, bb, cc, dd = cipher_round(a, b, c, d, ctr1[i, :], ctr2[i, :], ctr3[i, :], ctr4[i, :])
            out[:, 0, 4*i:(4*i)+4] = aa
            out[:, 1, 4*i:(4*i)+4] = bb
            out[:, 2, 4*i:(4*i)+4] = cc
            out[:, 3, 4*i:(4*i)+4] = dd
    else:
        for i in range(n):
            a = msg[:, :, 4*i]
            b = msg[:, :, 4*i+1]
            c = msg[:, :, 4*i+2]
            d = msg[:, :, 4*i+3]
            aa, bb, cc, dd = cipher_round(a, b, c, d, ctr1[i, :], ctr2[i, :], ctr3[i, :], ctr4[i, :])
            out[:, :, 4 * i] = aa
            out[:, :, 4 * i + 1] = bb
            out[:, :, 4 * i + 2] = cc
            out[:, :, 4 * i + 3] = dd

    return out

def bloc_decrypt(msg, ctr1, ctr2, ctr3, ctr4, s_mode):

    def decipher_round(a,b,c,d,ctr1,ctr2,ctr3,ctr4):
        aa = np.reshape(a, (1, 16), order='F')
        bb = np.reshape(b, (1, 16), order='F')
        cc = np.reshape(c, (1, 16), order='F')
        dd = np.reshape(d, (1, 16), order='F')
        aa = aa ^ dd
        dd = dd ^ ctr4
        dd = dd ^ cc
        cc = bb ^ cc
        cc = cc ^ ctr3
        bb = cc ^ bb
        bb = bb ^ ctr2
        bb = bb ^ aa
        dd = dd ^ aa
        aa = aa ^ ctr1
        aa = np.mod(np.reshape(aa, (4, 4), order='F'), 256)
        bb = np.mod(np.reshape(bb, (4, 4), order='F'), 256)
        cc = np.mod(np.reshape(cc, (4, 4), order='F'), 256)
        dd = np.mod(np.reshape(dd, (4, 4), order='F'), 256)
        return aa, bb, cc, dd

    x, y, z = msg.shape
    n = int(z/4)
    out = np.empty((x, y, z), dtype=np.uint8)

    if s_mode == 1:
        for i in range(n):
            a = msg[0, :, 4*i:(4*i)+4]
            b = msg[1, :, 4*i:(4*i)+4]
            c = msg[2, :, 4*i:(4*i)+4]
            d = msg[3, :, 4*i:(4*i)+4]
            aa, bb, cc, dd = decipher_round(a, b, c, d, ctr1[i, :], ctr2[i, :], ctr3[i, :], ctr4[i, :])
            out[0, :, 4*i:(4*i)+4] = aa
            out[1, :, 4*i:(4*i)+4] = bb
            out[2, :, 4*i:(4*i)+4] = cc
            out[3, :, 4*i:(4*i)+4] = dd
    elif s_mode == 2:
        for i in range(n):
            a = msg[:, 0, 4*i:(4*i)+4]
            b = msg[:, 1, 4*i:(4*i)+4]
            c = msg[:, 2, 4*i:(4*i)+4]
            d = msg[:, 3, 4*i:(4*i)+4]
            aa, bb, cc, dd = decipher_round(a, b, c, d, ctr1[i, :], ctr2[i, :], ctr3[i, :], ctr4[i, :])
            out[:, 0, 4*i:(4*i)+4] = aa
            out[:, 1, 4*i:(4*i)+4] = bb
            out[:, 2, 4*i:(4*i)+4] = cc
            out[:, 3, 4*i:(4*i)+4] = dd
    else:
        for i in range(n):
            a = msg[:, :, 4*i]
            b = msg[:, :, 4*i+1]
            c = msg[:, :, 4*i+2]
            d = msg[:, :, 4*i+3]
            aa, bb, cc, dd = decipher_round(a, b, c, d, ctr1[i, :], ctr2[i, :], ctr3[i, :], ctr4[i, :])
            out[:, :, 4*i] = aa
            out[:, :, 4*i+1] = bb
            out[:, :, 4*i+2] = cc
            out[:, :, 4*i+3] = dd

    return out


def block_rubik_90(state):
    out = np.empty((4, 4), dtype=int)

    out[0, 0] = state[3, 0]
    out[1, 0] = state[3, 1]
    out[2, 0] = state[3, 2]
    out[3, 0] = state[3, 3]

    out[0, 1] = state[2, 0]
    out[1, 1] = state[2, 1]
    out[2, 1] = state[2, 2]
    out[3, 1] = state[2, 3]

    out[0, 2] = state[1, 0]
    out[1, 2] = state[1, 1]
    out[2, 2] = state[1, 2]
    out[3, 2] = state[1, 3]

    out[0, 3] = state[0, 0]
    out[1, 3] = state[0, 1]
    out[2, 3] = state[0, 2]
    out[3, 3] = state[0, 3]

    return out

def block_rubik_270(state):

    out = np.empty((4, 4), dtype=int)

    out[0, 0] = state[0, 3]
    out[1, 0] = state[0, 2]
    out[2, 0] = state[0, 1]
    out[3, 0] = state[0, 0]

    out[0, 1] = state[1, 3]
    out[1, 1] = state[1, 2]
    out[2, 1] = state[1, 1]
    out[3, 1] = state[1, 0]

    out[0, 2] = state[2, 3]
    out[1, 2] = state[2, 2]
    out[2, 2] = state[2, 1]
    out[3, 2] = state[2, 0]

    out[0, 3] = state[3, 3]
    out[1, 3] = state[3, 2]
    out[2, 3] = state[3, 1]
    out[3, 3] = state[3, 0]

    return out

def rubik_shuffle(msg, mode):

    a, b, c = msg.shape
    out = np.copy(msg)
    if mode == 1:
        for i in range(int(c/4)):
            out[:, :, 4*i+1] = block_rubik_90(msg[:, :, 4*i+1])
    elif mode == 2:
        for i in range(int(c/4)):
            out[:, :, 4*i+2] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[:, :, 4*i+2]))))
    elif mode == 3:
        for i in range(int(c/4)):
            out[:, :, 4*i+3] = block_rubik_270(msg[:, :, 4*i+3])
    elif mode == 4:
        out[0, :, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[0, :, :]))))
    elif mode == 5:
        out[2, :, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[2, :, :]))))
    elif mode == 6:
        out[:, 1, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[:, 1, :]))))
    else:
        out[:, 3, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[:, 3, :]))))

    return out

def inv_rubik_shuffle(msg, mode):
    a, b, c = msg.shape
    out = np.copy(msg)
    if mode == 1:
        for i in range(int(c/4)):
            out[:, :, 4*i+1] = block_rubik_270(msg[:, :, 4*i+1])
    elif mode == 2:
        for i in range(int(c/4)):
            out[:, :, 4*i+2] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[:, :, 4*i+2]))))
    elif mode == 3:
        for i in range(int(c/4)):
            out[:, :, 4*i+3] = block_rubik_90(msg[:, :, 4*i+3])
    elif mode == 4:
        out[0, :, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[0, :, :]))))
    elif mode == 5:
        out[2, :, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[2, :, :]))))
    elif mode == 6:
        out[:, 1, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[:, 1, :]))))
    else:
        out[:, 3, :] = np.fliplr(np.transpose(np.fliplr(np.transpose(msg[:, 3, :]))))

    return out


def sudoku_shuffle(msg, idx, mtrix, s_box, counter, s_mode):
    n = 64
    mtrix = np.array(mtrix)
    x, y, z = msg.shape
    msg = msg.reshape((1, x*y*z), order='F')
    out = np.empty((x, y, z), dtype=int)
    for i in range(int(x*y*z/64)):
        temp = msg[0, 64*i:(64*i)+64]

        if counter == 1:
            temp = message_substitution(temp, s_box, s_mode)

        temp = temp[mtrix[np.mod(idx + i, n), :].tolist()]
        temp = temp[mtrix[:, np.mod(idx + i, n)].tolist()]
        out[:, :, 4*i:(4*i)+4] = np.reshape(temp, (4, 4, 4), order='F')
    return out


def inv_sudoku_shuffle(msg, idx, inv_mtrix_r, inv_mtrix_c, inv_s_box, counter, s_mode):
    n = 64
        
    x, y, z = msg.shape
    msg = msg.reshape((1, x * y * z), order='F')
    out = np.empty((x, y, z), dtype=int)
    for i in range(int(x*y*z/64)):
        temp = msg[0, 64*i:(64*i)+64]
        temp = temp[inv_mtrix_c[:, np.mod(idx + i, n)].tolist()]
        temp = temp[inv_mtrix_r[np.mod(idx + i, n), :].tolist()]

        if counter == 1:
            temp = message_substitution(temp, inv_s_box, s_mode)
            
        out[:, :, 4*i:(4*i)+4] = np.reshape(temp, (4, 4, 4), order='F')
    return out


def message_reshaping(msg):

    def pkcs7_pad(msg):
        def padding(msg):
            dimn = msg.shape
            dimn = np.array(dimn, dtype=int).tolist()
            dmn_copy = dimn[0]
            lngth = np.prod(dimn)
            lngth_copy = lngth
            idx = 0
            while np.mod(lngth, 64) != 0:
                dimn[0] = dmn_copy + idx
                lngth = np.prod(dimn)
                idx = idx + 1

            return lngth - lngth_copy, idx - 1

        p, idxx = padding(msg)
        pad = int(str(p), 16)
        pad = np.ones((p, 1), dtype=int) * pad
        msg = msg.reshape((np.prod(msg.shape), 1), order='F')
        out = np.concatenate((msg, pad))

        return out, p, idxx


    dimn = msg.shape
    msg, p, idxx = pkcs7_pad(msg)
    l = int(np.prod(msg.shape))
    msg = np.reshape(msg, (4, 4, int(4 * (l / 64))), order='F')
    if idxx == -1:
        idxx = 0

    return msg, p, int(l/64), dimn, idxx


def message_substitution(temp, s_box, s_mode):

    if s_mode == 1:
        temp[0] = s_box[int(temp[0])]
        temp[1] = s_box[int(temp[1])]
        temp[2] = s_box[int(temp[2])]
        temp[3] = s_box[int(temp[3])]
        temp[12] = s_box[int(temp[12])]
        temp[13] = s_box[int(temp[13])]
        temp[14] = s_box[int(temp[14])]
        temp[15] = s_box[int(temp[15])]
        temp[16] = s_box[int(temp[16])]
        temp[17] = s_box[int(temp[17])]
        temp[18] = s_box[int(temp[18])]
        temp[19] = s_box[int(temp[19])]
        temp[28] = s_box[int(temp[28])]
        temp[29] = s_box[int(temp[29])]
        temp[30] = s_box[int(temp[30])]
        temp[31] = s_box[int(temp[31])]
        temp[32] = s_box[int(temp[32])]
        temp[33] = s_box[int(temp[33])]
        temp[34] = s_box[int(temp[34])]
        temp[35] = s_box[int(temp[35])]
        temp[44] = s_box[int(temp[44])]
        temp[45] = s_box[int(temp[45])]
        temp[46] = s_box[int(temp[46])]
        temp[47] = s_box[int(temp[47])]
        temp[48] = s_box[int(temp[48])]
        temp[49] = s_box[int(temp[49])]
        temp[50] = s_box[int(temp[50])]
        temp[51] = s_box[int(temp[51])]
        temp[60] = s_box[int(temp[60])]
        temp[61] = s_box[int(temp[61])]
        temp[62] = s_box[int(temp[62])]
        temp[63] = s_box[int(temp[63])]

    elif s_mode == 2:
        temp[0] = s_box[int(temp[0])]
        temp[4] = s_box[int(temp[4])]
        temp[8] = s_box[int(temp[8])]
        temp[12] = s_box[int(temp[12])]
        temp[3] = s_box[int(temp[3])]
        temp[7] = s_box[int(temp[7])]
        temp[11] = s_box[int(temp[11])]
        temp[15] = s_box[int(temp[15])]
        temp[16] = s_box[int(temp[16])]
        temp[20] = s_box[int(temp[20])]
        temp[24] = s_box[int(temp[24])]
        temp[28] = s_box[int(temp[28])]
        temp[19] = s_box[int(temp[19])]
        temp[23] = s_box[int(temp[23])]
        temp[27] = s_box[int(temp[27])]
        temp[31] = s_box[int(temp[31])]
        temp[32] = s_box[int(temp[32])]
        temp[36] = s_box[int(temp[36])]
        temp[40] = s_box[int(temp[40])]
        temp[44] = s_box[int(temp[44])]
        temp[35] = s_box[int(temp[35])]
        temp[39] = s_box[int(temp[39])]
        temp[43] = s_box[int(temp[43])]
        temp[47] = s_box[int(temp[47])]
        temp[48] = s_box[int(temp[48])]
        temp[52] = s_box[int(temp[52])]
        temp[56] = s_box[int(temp[56])]
        temp[60] = s_box[int(temp[60])]
        temp[51] = s_box[int(temp[51])]
        temp[55] = s_box[int(temp[55])]
        temp[59] = s_box[int(temp[59])]
        temp[63] = s_box[int(temp[63])]

    else:
        temp[0] = s_box[int(temp[0])]
        temp[1] = s_box[int(temp[1])]
        temp[2] = s_box[int(temp[2])]
        temp[3] = s_box[int(temp[3])]
        temp[4] = s_box[int(temp[4])]
        temp[5] = s_box[int(temp[5])]
        temp[6] = s_box[int(temp[6])]
        temp[7] = s_box[int(temp[7])]
        temp[8] = s_box[int(temp[8])]
        temp[9] = s_box[int(temp[9])]
        temp[10] = s_box[int(temp[10])]
        temp[11] = s_box[int(temp[11])]
        temp[12] = s_box[int(temp[12])]
        temp[13] = s_box[int(temp[13])]
        temp[14] = s_box[int(temp[14])]
        temp[15] = s_box[int(temp[15])]
        temp[48] = s_box[int(temp[48])]
        temp[49] = s_box[int(temp[49])]
        temp[50] = s_box[int(temp[50])]
        temp[51] = s_box[int(temp[51])]
        temp[52] = s_box[int(temp[52])]
        temp[53] = s_box[int(temp[53])]
        temp[54] = s_box[int(temp[54])]
        temp[55] = s_box[int(temp[55])]
        temp[56] = s_box[int(temp[56])]
        temp[57] = s_box[int(temp[57])]
        temp[58] = s_box[int(temp[58])]
        temp[59] = s_box[int(temp[59])]
        temp[60] = s_box[int(temp[60])]
        temp[61] = s_box[int(temp[61])]
        temp[62] = s_box[int(temp[62])]
        temp[63] = s_box[int(temp[63])]

    return temp
        

def full_encryption(msg, n, ctr1, ctr2, ctr3, ctr4, sbox1, sbox2, k_dep_c, enc_times):
    # n is the number of 64-bytes blocks (i.e., 4x4x4 cubes)
    
    #for k in range(7):
        #msg = rubik_shuffle(msg, mode=k + 1)

    msg = rubik_shuffle(msg, mode=1)
    msg = rubik_shuffle(msg, mode=2)
    msg = rubik_shuffle(msg, mode=3)
    msg = rubik_shuffle(msg, mode=4)
    msg = rubik_shuffle(msg, mode=5)
    msg = rubik_shuffle(msg, mode=6)
    msg = rubik_shuffle(msg, mode=7)

    sudoku_index1 = [0]*enc_times
    sudoku_index2 = [0]*enc_times
    s_mode = [0]*enc_times
    sub_ctr = None

    for i in range(enc_times):
        if i == 0:
            sub_ctr = 1
        else:
            sub_ctr = 0
            
        sudoku_index1[i] = random.randrange(0, 31) + k_dep_c
        sudoku_index2[i] = random.randrange(0, 31) + k_dep_c
        s_mode[i] = random.choice([1,2,3])

        msg = sudoku_shuffle(msg, sudoku_index1[i], m_64, sbox1, sub_ctr, s_mode[i])

        msg = bloc_encrypt(msg, ctr1[n*i:(n*i)+(n)], ctr2[n*i:(n*i)+(n)], ctr3[n*i:(n*i)+(n)],
                           ctr4[n*i:(n*i)+(n)], s_mode[i])

        msg = sudoku_shuffle(msg, sudoku_index2[i], m_64, sbox1, sub_ctr, s_mode[i])


    return msg, sudoku_index1, sudoku_index2, s_mode


def partial_encryption(msg, n, ctr1, ctr2, ctr3, ctr4, sbox1, sbox2, k_dep_c, enc_times):
    # n is the number of 64-bytes blocks (i.e., 4x4x4 cubes)
    
    #for k in range(7):
        #msg = rubik_shuffle(msg, mode=k + 1)

    
    msg = rubik_shuffle(msg, mode=1)
    msg = rubik_shuffle(msg, mode=2)
    msg = rubik_shuffle(msg, mode=3)
    msg = rubik_shuffle(msg, mode=4)
    msg = rubik_shuffle(msg, mode=5)
    msg = rubik_shuffle(msg, mode=6)
    msg = rubik_shuffle(msg, mode=7)

    sudoku_index1 = [0]*enc_times
    sudoku_index2 = [0]*enc_times
    s_mode = [0]*enc_times
    sub_ctr = None

    for i in range(enc_times):
        if i == 0:
            sub_ctr = 1
        else:
            sub_ctr = 0
            
        sudoku_index1[i] = random.randrange(0, 31) + k_dep_c
        sudoku_index2[i] = random.randrange(0, 31) + k_dep_c
        s_mode[i] = random.choice([1,2,3])

        msg = sudoku_shuffle(msg, sudoku_index1[i], m_64, sbox1, sub_ctr, s_mode[i])

        #msg = sudoku_shuffle(msg, sudoku_index2[i], m_64, sbox1, sub_ctr, s_mode[i])


    return msg, sudoku_index1, sudoku_index2, s_mode


def full_decryption(msg, n, ctr1, ctr2, ctr3, ctr4, inv_sbox1, inv_sbox2,
                    sudoku_index1, sudoku_index2, s_mode, dec_times):

    for i in range(dec_times)[::-1]:
        if i == 0:
            sub_ctr = 1
        else:
            sub_ctr = 0

        msg = inv_sudoku_shuffle(msg, sudoku_index2[i], inv_m64_r, inv_m64_c, inv_sbox1, sub_ctr, s_mode[i])
        
        msg = bloc_decrypt(msg, ctr1[n*i:(n*i)+(n)], ctr2[n*i:(n*i)+(n)], ctr3[n*i:(n*i)+(n)],
                           ctr4[n*i:(n*i)+(n)], s_mode[i])

        msg = inv_sudoku_shuffle(msg, sudoku_index1[i], inv_m64_r, inv_m64_c, inv_sbox1, sub_ctr, s_mode[i])

    msg = inv_rubik_shuffle(msg, mode=7)
    msg = inv_rubik_shuffle(msg, mode=6)
    msg = inv_rubik_shuffle(msg, mode=5)
    msg = inv_rubik_shuffle(msg, mode=4)
    msg = inv_rubik_shuffle(msg, mode=3)
    msg = inv_rubik_shuffle(msg, mode=2)
    msg = inv_rubik_shuffle(msg, mode=1)

    #for q in range(7)[::-1]:
    #    msg = inv_rubik_shuffle(msg, mode=q + 1)


    return msg

def partial_decryption(msg, n, ctr1, ctr2, ctr3, ctr4, inv_sbox1, inv_sbox2,
                    sudoku_index1, sudoku_index2, s_mode, dec_times):

    for i in range(dec_times)[::-1]:
        if i == 0:
            sub_ctr = 1
        else:
            sub_ctr = 0

        #msg = inv_sudoku_shuffle(msg, sudoku_index2[i], inv_m64_r, inv_m64_c, inv_sbox1, sub_ctr, s_mode[i])
        
        msg = inv_sudoku_shuffle(msg, sudoku_index1[i], inv_m64_r, inv_m64_c, inv_sbox1, sub_ctr, s_mode[i])

    
    msg = inv_rubik_shuffle(msg, mode=7)
    msg = inv_rubik_shuffle(msg, mode=6)
    msg = inv_rubik_shuffle(msg, mode=5)
    msg = inv_rubik_shuffle(msg, mode=4)
    msg = inv_rubik_shuffle(msg, mode=3)
    msg = inv_rubik_shuffle(msg, mode=2)
    msg = inv_rubik_shuffle(msg, mode=1)
    

    #for q in range(7)[::-1]:
    #    msg = inv_rubik_shuffle(msg, mode=q + 1)


    return msg



