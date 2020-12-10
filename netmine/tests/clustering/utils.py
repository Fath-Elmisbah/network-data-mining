def convert_to_binary(num, length=8):
    """
    Converts a number to binary

    :param int num: the number to be converted
    :param int length: the length of the produced binary output. The output will be padded with zeros to satisfy
    this requirement. This parameter should be set carefully by the user of the function because if it is too low
    to represent the number, the parameter will be ignored and a representation of the full binary number is
    returned.
    :return str: a string representation of the binary number
    """
    binary = bin(num)[2:]
    padding_length = length - len(binary)
    binary = str(0)*padding_length + binary
    return binary
