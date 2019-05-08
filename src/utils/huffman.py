import heapq
import os
import numpy as np

class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if(other == None):
            return False
        if(not isinstance(other, HeapNode)):
            return False
        return self.freq == other.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    # functions for compression:

    # def make_frequency_dict(self, text):
    def make_frequency_dict(self, nparray):
        # get 1d np array
        frequency = {}
        for number in nparray:
            if not number in frequency:
                frequency[number] = 0
            frequency[number] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if(root == None):
            return
        if(root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)


    def get_encoded_bits(self, nparray):
        encoded_bits = ""
        for number in nparray:
            encoded_bits += self.codes[number]
        return encoded_bits


    def pad_encoded_bits(self, encoded_bits):
        extra_padding = 8 - len(encoded_bits) % 8
        for i in range(extra_padding):
            encoded_bits += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_bits += padded_info
        return encoded_bits


    def get_byte_array(self, padded_encoded_bits):
        if(len(padded_encoded_bits) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_bits), 8):
            byte = padded_encoded_bits[i:i+8]
            b.append(int(byte, 2))
        return bytes(b)

    # def encode_mapping(self):
    #     # encode the reverse map
    #     # self.reverse_mapping
    #     for bits, number in self.reverse_mapping.items():

    #     return info

    def compress(self, nparray):
        # 2d nparray
        frequency = self.make_frequency_dict(nparray.flatten())
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        encoded_bits = self.get_encoded_bits(nparray.flatten())
        padded_encoded_bits = self.pad_encoded_bits(encoded_bits)

        b = self.get_byte_array(padded_encoded_bits)
        info = self.encode_mapping()

        print("Compressed")
        return b, info


    """ functions for decompression: """

    def decode_mapping(self):
        pass

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:] 
        encoded_text = padded_encoded_text[:-1*extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if(current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text


    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + "_decompressed" + ".txt"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while(byte != ""):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_text = self.decode_text(encoded_text)
            
            output.write(decompressed_text)

        print("Decompressed")
        return output_path