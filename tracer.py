import numpy as np
from PIL import Image

class Tracer:
    def __init__():
        pass

if __name__ == "__main__":
    '''
    r = np.linspace(0, 255, num=256, dtype=np.uint8)
    print(r.shape)
    g = np.zeros(256, dtype=np.uint8)
    print(g.shape)
    b = np.linspace(255, 0, num=256, dtype=np.uint8)
    print(b.shape)
    rgb = np.column_stack((r,g,b))
    print(rgb.shape)
    rgb = rgb.reshape((16,16,3))
    print(rgb.shape)
    im = Image.fromarray(rgb)
    im.save("Renders/helloworld.png")
    '''
    # equivlent to above
    r = np.linspace(0, 255, num=256, dtype=np.uint8).reshape((16,16))
    print(r.shape)
    g = np.zeros(256, dtype=np.uint8).reshape((16,16))
    print(g.shape)
    b = np.linspace(255, 0, num=256, dtype=np.uint8).reshape((16,16))
    print(b.shape)
    rgb = np.dstack((r,g,b))
    print(rgb.shape)
    im = Image.fromarray(rgb)
    im.save("Renders/helloworld.png")