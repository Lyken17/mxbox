import mxnet as mx
from mxbox import transforms

def avaliable_devices_count(maxium=16):
    """
    Detect how many gpus are available on this machine.
    :param maxium: the maxium range that will be checked, default to 16, twice of 8. Should be enough on most servers.
    :return:
    """
    # FIXME: cuda level implementation wanted

    for i in range(maxium):
        try:
            mx.nd.array([0], ctx=mx.gpu(i))
        except:
            return i
    return i


def ndarray2image():
    raise NotImplementedError

if __name__ == "__main__":
    print (avaliable_devices_count(16))