import base64
import pickle

import lz4.frame


def compress_data(data):
    data = pickle.dumps(data)
    data = lz4.frame.compress(data)
    data = base64.b64encode(data).decode("ascii")
    return data


def decompress_data(data):
    data = base64.b64decode(data)
    data = lz4.frame.decompress(data)
    data = pickle.loads(data)
    return data
