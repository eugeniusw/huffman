from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from collections import deque
sys.setrecursionlimit(2000)

class HuffmanTreeCell:
    def __init__(self):
        self.data = None
        self.freq = None
        self.left = None
        self.right = None

class PriorityQueue:
    def __init__(self):
        self.queue = []
    def push(self, huffmanTreeCell):
        self.queue.append(huffmanTreeCell)
        self.queue.sort(key=lambda h: h.freq, reverse=True)
    def pop(self):
        return self.queue.pop()
    def top(self):
        return self.queue[-1]
    def size(self):
        return len(self.queue)

def generateEncoding(root, code):
    if root.left == None and root.right == None:
        yield (root.data, code)
        return
    yield from generateEncoding(root.left, code+"0")
    yield from generateEncoding(root.right, code+"1")

def generateCell(root, encoded, value):
    if len(encoded) == 0:
        root.data = value
        return
    s = encoded.pop()
    if s == '0' and root.left == None:
        new_cell = HuffmanTreeCell()
        root.left = new_cell
        generateCell(root.left, encoded, value)
    elif s == '0' and root.left != None:
        generateCell(root.left, encoded, value)
    elif s == '1' and root.right == None:
        new_cell = HuffmanTreeCell()
        root.right = new_cell
        generateCell(root.right, encoded, value)
    elif s == '1' and root.right != None:
        generateCell(root.right, encoded, value)

def generateHuffmanTreeFromEncoding(huffmanEncoding, root=None):
    """ huffmanEncoding should be in format [(value_1, encoded_1), ... , (value_n, encoded_n)]"""
    if root == None:
        root = HuffmanTreeCell()
    for value, encoded in huffmanEncoding:
        encoded = list(encoded)
        encoded.reverse()
        generateCell(root, encoded, value)
    return root

def decodeHuffmanEncoding(huffmanEncoding, encoded_data):
    """ huffmanEncoding should be in format [(value_1, encoded_1), ... , (value_n, encoded_n)]"""
    root = generateHuffmanTreeFromEncoding(huffmanEncoding)
    decoded_data = ''
    encoded_data = list(encoded_data)
    encoded_data.append('0') # bonus bit to prevent array empty lookup
    encoded_data.reverse()
    def _startDecode(root, encoded_data):
        s = encoded_data[-1]
        if s == '0' and root.left != None:
            encoded_data.pop()
            return _startDecode(root.left, encoded_data)
        elif s == '0' and root.left == None:
            return encoded_data, root.data
        elif s == '1' and root.right != None:
            encoded_data.pop()
            return _startDecode(root.right, encoded_data)
        elif s == '1' and root.right == None:
            return encoded_data, root.data
    while len(encoded_data) > 1:
        encoded_data, decoded = _startDecode(root, encoded_data)
        decoded_data += decoded
    return decoded_data

def huffmanImageDecoder(encoded_value, n_dim=None): #huffmanEncoding, encoded_data):
    def decodeHeader(encoded_value):
        total_last_pad = int(encoded_value[0:8], 2) # base 2 of first byte
        width = int(encoded_value[8:40], 2) # 32 bit length 
        height = int(encoded_value[40:72], 2) # 32 bit length
        total_map_data = int(encoded_value[72:80], 2) + 1 # because 0 means there is 1 byte data. see wrap function!
        # map data
        huffmanEncoding = []
        i = 80
        count = 0 
        while 1:
            ori_val = int(encoded_value[i:i+8],2)
            total_pad = int(encoded_value[i+8:i+16],2)
            byte_length = int(encoded_value[i+16:i+24],2) 
            i = i + 24
            pad_enc_val = encoded_value[i: i+byte_length*8]
            enc_val = pad_enc_val[total_pad:]
            huffmanEncoding.append((ori_val, enc_val))
            i = i + (byte_length*8)
            count += 1
            if count == total_map_data:
                break
        # body/encoded data without last pad
        encoded_data = encoded_value[i:-total_last_pad]
        return width, height, huffmanEncoding, encoded_data

    width, height, huffmanEncoding, encoded_data = decodeHeader(encoded_value)
    # root_time_start = time.time()
    root = generateHuffmanTreeFromEncoding(huffmanEncoding)
    # root_time_end = time.time()
    # print(f"Root generated in {root_time_end-root_time_start} seconds.")
    # body_time_start = time.time()
    decoded_data = deque()
    encoded_data = deque(encoded_data)
    encoded_data.append('0') # bonus bit to prevent array empty lookup
    def _startDecode(root, encoded_data):
        s = encoded_data[0]
        if s == '0' and root.left != None:
            encoded_data.popleft()
            return _startDecode(root.left, encoded_data)
        elif s == '0' and root.left == None:
            return root.data
        elif s == '1' and root.right != None:
            encoded_data.popleft()
            return _startDecode(root.right, encoded_data)
        elif s == '1' and root.right == None:
            return root.data
    try:
        while 1:
            decoded = _startDecode(root, encoded_data)
            decoded_data.append(decoded)
    except Exception as e:
        # print("complete",e)
        pass
    # body_time_end = time.time()
    # print(f"Body decoded in {body_time_end-body_time_start} seconds.")
    # convert_time_start = time.time()
    arr_img = np.array(decoded_data)
    if n_dim:
        arr_img = arr_img.reshape(height, width, n_dim)
    else:
        arr_img = arr_img.reshape(height, width)
    # convert_time_end = time.time()
    # print(f"convert to numpy takes {convert_time_end-convert_time_start} seconds.")
    return arr_img

def huffman(value_frequency):
    # add data in priority queue
    queue = PriorityQueue()
    for data, freq in value_frequency:
        # create tree cell
        tree_cell = HuffmanTreeCell()
        tree_cell.data = data
        tree_cell.freq = freq
        tree_cell.left = None
        tree_cell.right = None
        # add tree_cell to queue
        queue.push(tree_cell)
    # create a root cell
    root = None
    # extract two minimum value from queue
    # until it size reduce to 1
    while queue.size() > 1:
        # first min extract
        x = queue.pop()
        # second min extract
        y = queue.pop()
        # sum the freq to f cell
        f = HuffmanTreeCell()
        f.data = '-'
        f.freq = x.freq + y.freq
        f.left = x
        f.right = y
        root = f
        queue.push(f)
    # traversing the tree
    try:
        yield from generateEncoding(root, "")
    except:
        root = queue.pop()
        yield (root.data,"0")

def toBytes(data):
    for i in range(0, len(data), 8):
        yield bytes([int(data[i:i+8], 2)])

def wrapEncodedData(huffmanMap, encoded_data, width, height):
    """ 
        header represented in:
            [total last bit pad from encoded data] (8 bit)
            [image width] (32 bit)
            [image height] (32 bit)
            [total map data] (8 bit)
            [map data]
                map data represented in
                [original value (0-255)] (8 bit)
                [total pad of represented encoded value] ( 8 bit )
                [byte length of encoded value] (8 bit)
                [encoded value] + [pad] (x bit)
                ...
        body (encoded data)
                ...
                +[last_pad]
    """
    encoded_data = ''.join(list(encoded_data)) # change this later to optimize memory
    total_last_bit_pad = 8-len(encoded_data)%8
    b_total_last_bit_pad = '{0:08b}'.format(total_last_bit_pad)
    width = '{0:032b}'.format(width)
    height = '{0:032b}'.format(height)
    total_map_data = '{0:08b}'.format(len(huffmanMap)-1) # need to save 1 byte so byte_length=0 means that there is 1 byte
    header = b_total_last_bit_pad + width + height + total_map_data
    map_data = ''
    for ori_val, enc_val in huffmanMap:
       b_ori_val = '{0:08b}'.format(ori_val)
       total_pad = 8-len(enc_val)%8
       b_total_pad = '{0:08b}'.format(total_pad)
       b_enc_val = total_pad*'0' + enc_val
       byte_length = '{0:08b}'.format(int(len(b_enc_val)/8)) 
       map_data = map_data + b_ori_val + b_total_pad + byte_length + b_enc_val
    header = header + map_data
    last_bit_pad = total_last_bit_pad * '0'
    wrapped_data = header + encoded_data + last_bit_pad
    return wrapped_data

def encodePixelValue(huffmanMap, img_array):
    # create dict for easier lookup
    huffmanMapDict = {key_value[0]: key_value[1] for key_value in huffmanMap}
    # encode
    encoded_px = (huffmanMapDict[px] for px in img_array)
    return encoded_px   
        
def main(argv):
    file_name = argv[1]
    im = Image.open(f"{file_name}.jpg")
    im_ar = np.array(im)
    # convert to gray 
    # im_ar = (im_ar[:,:,0] + im_ar[:,:,1] + im_ar[:,:,2])/3
    # im_ar = im_ar.astype('uint8')

    # create array 100x100 with value 255
    # im_ar = np.zeros([100,100], dtype='uint8')
    # im_ar.fill(255)
    try:
        height, width, dim = im_ar.shape
    except ValueError:
        height, width = im_ar.shape
        dim = None

    # save raw/generated image
    raw_img = Image.fromarray(im_ar)
    raw_img.save(f"{file_name}_out.tiff") # use tiff value for raw / no compression
    raw_img.save(f"{file_name}_out.jpg") # use jpg for compression comparison later
    
    # measuring encode algorithm
    encode_start_time = time.time()

    # calculate total pixel and pixel freq
    im_grayflat = im_ar.ravel()
    hist = np.bincount(im_grayflat)
    prob = hist/np.sum(hist)
    
    # get freq > 0
    value_freq = ((v, f) for v, f in enumerate(prob) if f > 0)
    huffmanMap = list(huffman(value_freq))

    # encode image pixel value
    encoded_pixel = encodePixelValue(huffmanMap, im_grayflat)

    # let's write it to file
    bytes_generator = toBytes(wrapEncodedData(huffmanMap, encoded_pixel, width, height))
    with open(f"{file_name}.huff", "wb") as f:
        for data in bytes_generator:
            f.write(data)
    
    # measuring encode algorithm
    encode_stop_time = time.time()
    print(f"completed huffman encode compression in {encode_stop_time - encode_start_time} seconds.")

    # measuring decode algorithm
    decode_start_time = time.time()

    # let's read it from file
    with open(f"{file_name}.huff", "rb") as f:
        enc_img = f.read()
    
    # decode it
    b_enc_img = ''.join(map(lambda x: '{:08b}'.format(x), enc_img))
    arr_img = huffmanImageDecoder(b_enc_img,dim)
    arr_img = arr_img.astype('uint8')

    # measuring decode algorithm
    decode_stop_time = time.time()
    print(f"completed huffman compression decode in {decode_stop_time - decode_start_time} seconds.")

    # measure size of files
    print("==== files sizes ====")
    raw_img_size = os.path.getsize(f"{file_name}_out.tiff")
    jpg_img_size = os.path.getsize(f"{file_name}_out.jpg")
    huffman_img_size = os.path.getsize(f"{file_name}.huff")
    compression_ratio1 = 100 - (huffman_img_size/raw_img_size)*100
    compression_ratio2 = 100 - (huffman_img_size/jpg_img_size)*100
    print(f"raw image size (TIFF) : {raw_img_size} bytes.")
    print(f"compressed image size (JPG) : {jpg_img_size} bytes.")
    print(f"compressed image size (HUFF) : {huffman_img_size} bytes.")
    print(f"huffman compression ratio vs raw: {compression_ratio1} %")
    print(f"huffman compression ratio vs jpg compression: {compression_ratio2} %")

    # show huffman map
    print("==== Huffman Map ====")
    huffmanMap.sort(key = lambda x: x[0])
    [print(f"{key_value[0]}\t{key_value[1]}") for key_value in huffmanMap]
    # compare decoded and raw 
    raw_image = Image.open(f"{file_name}_out.tiff") 
    raw_image = np.array(raw_image, dtype='uint8') 
    arr_img = arr_img.astype('uint8')
    fig, ax = plt.subplots(1,2)
    if dim:
        ax[0].imshow(raw_img)
        ax[0].set_title("RAW IMAGE")
        ax[1].imshow(arr_img)
        ax[1].set_title("DECODED IMAGE")
    else:
        ax[0].imshow(raw_img, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title("RAW IMAGE")
        ax[1].imshow(arr_img, cmap='gray', vmin=0, vmax=255)
        ax[1].set_title("DECODED IMAGE")
    plt.show()
    
main(sys.argv)
