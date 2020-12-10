import numpy as np

def encode_2d_in_1d_array(array, max_val=10000):
    """
    converts a 2d array into a numeric 1d array by encoding the numbers

    The array used should neither have nor be expected to have any value equals to or greater than max_val for
    the encoding and decoding to occure accurately. decodeing is made by decode_1d_to_2d_array

    :param ndarray array: the 2d ndarray to be encoded.
    :param int max_val: the maximum value the array is expected to have during the period it is encoded.
    :return ndarray: a 1d ndarray that contains an encoded version of the array
    """
    return array[:,0] * max_val + array[:,1]

def decode_1d_to_2d_array(array, max_val=10000):
    """
    converts a 1d array encoded by encode_2d_in_1d_array into a numeric 2d array

    The array used should neither have nor be expected to have any value equals to or greater than max_val fot
    the encoding and decoding to occure accurately.

    :param ndarray array: the 1d ndarray be decoded.
    :param int max_val: the maximum value the array is expected to have during the period it is encoded.
    :return ndarray: a 2d ndarray that represents the array before encoding
    """

    return np.concatenate((array.reshape((-1,1))//max_val,array.reshape((-1,1))%max_val),1)