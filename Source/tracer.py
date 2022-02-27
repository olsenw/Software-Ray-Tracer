import numpy as np
from PIL import Image

class ParametricSphere:
    def __init__(self, center=(0,0,0), radius=1, color=(0,0,0), specular=10):
        self.center = center
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
    
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
        t2 = (-b - np.sqrt(d)) / (2 * a)
        return t1, t2

class RayTracer:
    def __init__(self, canvas=(400,300), viewport=(1,1,1), camera=(0,0,0), renderDistance=(1,float('inf'))):
        self.canvas = canvas
        self.viewport = viewport
        self.camera = np.array(camera)
        self.renderDistance = renderDistance
    
    def __lighting(self, point, normal, lights, ray, specular):
        intensity = 0.
        for a in lights[0]:
            intensity += a
        def calc(i, d):
            ans = 0
            # diffuse
            # make sure not lighting "behind" object
            t = np.dot(normal, d)
            if t > 0:
                ans += i * t / (np.linalg.norm(normal) * np.linalg.norm(d))
            # specular
            if specular >= 0:
                r = 2 * normal * np.dot(normal, d) - d
                rv = np.dot(r, ray)
                if rv > 0:
                    ans += i * (rv / (np.linalg.norm(r) * np.linalg.norm(ray))) ** specular
            return ans
        for i, d in lights[1]:
            intensity += calc(i, d)
        for i, d in lights[1]:
            intensity += calc(i, np.subtract(d, point))
        return intensity

    def __trace(self, cx, cy, scene, lights):
        vx = (cx - self.canvas[0] / 2) * self.viewport[0] / self.canvas[0]
        vy = (-cy + self.canvas[1] / 2) * self.viewport[1] / self.canvas[1]
        closest = (float('inf'), None)
        direction = np.array((vx, vy, 1))
        for s in scene:
            t1, t2 = s.intersections(self.camera, direction)
            if self.renderDistance[0] <= t1 <= self.renderDistance[1] and t1 < closest[0]:
                closest = (t1, s)
            if self.renderDistance[0] <= t2 <= self.renderDistance[1] and t2 < closest[0]:
                closest = (t2, s)
        if not closest[1]:
            return (0,0,0)
        point = self.camera + closest[0] * direction
        s = np.subtract(point, closest[1].center)
        normal = s / np.linalg.norm(s)
        color = closest[1].color * self.__lighting(point, normal, lights, np.subtract(self.camera, direction), closest[1].specular)
        return np.clip(color, 0, 255).astype(np.uint8)

    def render(self, scene, lights):
        # note that this will be [y][x][channel] when access/update
        canvas = np.zeros((self.canvas[1], self.canvas[0], 3), dtype=np.uint8)
        for y in range(self.canvas[1]):
            for x in range(self.canvas[0]):
                canvas[y,x] = np.array(self.__trace(x, y, scene, lights), dtype=np.uint8)
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
        ParametricSphere((0,-1,3), 1, (255,0,0), 500),
        ParametricSphere((2,0,4), 1, (0,0,255), 500),
        ParametricSphere((-2,0,4), 1, (0,255,0), 10),
        ParametricSphere((0,-5001,0), 5000, (255,255,0), 1000), # yellow
        ]
    lights = [
        [0.2], # ambient (intensity)
        [(0.2, (1,4,4))], # directional (intensity, (direction))
        [(0.6, (2,1,0))], # point (intensity, (location))
    ]
    # RayTracer((40,40)).render(scene, lights).save("Renders/canvas.png")
    RayTracer((400,400)).render(scene, lights).save("Renders/canvas.png")
