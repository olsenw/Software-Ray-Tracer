import numpy as np
from PIL import Image

class ParametricSphere:
    def __init__(self, center=(0,0,0), radius=1, color=(0,0,0)):
        self.center = center
        self.radius = radius
        self.color = color
    
    def intersections(self, origin, direction):
        offset = np.subtract(origin, self.center)
        # parameters for a t^2 + b t + c = 0
        a = np.dot(direction, direction)
        b = 2 * np.dot(offset, direction)
        c = np.dot(offset, offset) - self.radius * self.radius
        # discriminant 
        d = b * b - 4 * a * c
        # zero solutions (no intersections)
        if d < 0:
            return float('inf'), float('inf')
        # one (touch sphere) or two solutions (enter exit)
        t1 = (-b + np.sqrt(d)) / (2 * a)
        t2 = (-b + np.sqrt(d)) / (2 * a)
        return t1, t2

class RayTracer:
    def __init__(self, canvas=(400,300), viewport=(1,1,1), camera=(0,0,0), renderDistance=(1,float('inf'))):
        self.canvas = canvas
        self.viewport = viewport
        self.camera = camera
        self.renderDistance = renderDistance
    
    def trace(self, cx, cy, scene):
        vx = (cx - self.canvas[0] / 2) * self.viewport[0] / self.canvas[0]
        vy = (-cy + self.canvas[1] / 2) * self.viewport[1] / self.canvas[1]
        closest = (float('inf'), None)
        for s in scene:
            t1, t2 = s.intersections(self.camera, (vx, vy, 1))
            if self.renderDistance[0] <= t1 <= self.renderDistance[1] and t1 < closest[0]:
                closest = (t1, s)
            if self.renderDistance[0] <= t2 <= self.renderDistance[1] and t2 < closest[0]:
                closest = (t2, s)
        return closest[1].color if closest[1] else (0,0,0)

    def render(self, scene):
        # note that this will be [y][x][channel] when access/update
        canvas = np.zeros((self.canvas[1], self.canvas[0], 3), dtype=np.uint8)
        for y in range(self.canvas[1]):
            for x in range(self.canvas[0]):
                canvas[y,x] = np.array(self.trace(x, y, scene), dtype=np.uint8)
        return Image.fromarray(canvas)

def test_pillow_numpy():
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

def test_assign_color():
    # note that this will be [y][x][channel] when access/update
    canvas = np.zeros((600,800,3), dtype=np.uint8)
    print(canvas[300,400])
    canvas[290:310,390:410] = (255,0, 126)
    print(canvas[300,400])
    Image.fromarray(canvas).save("Renders/canvas.png")

if __name__ == "__main__":
    scene = [
        ParametricSphere((0,-1,3), 1, (255,0,0)),
        ParametricSphere((2,0,4), 1, (0,0,255)),
        ParametricSphere((-2,0,4), 1, (0,255,0))
        ]
    RayTracer((400,400)).render(scene).save("Renders/canvas.png")
