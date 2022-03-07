import numpy as np
from PIL import Image
from math import sqrt
from abc import ABC, abstractmethod

# handy utility functions for working with vectors (avoids numpy...)
class Utility:
    @staticmethod
    def add(x,y):
        if len(x) != len(y):
            raise Exception("Mismatched Vector Lengths")
        return tuple(i + j for i,j in zip(x,y))

    @staticmethod
    def subtract(x,y):
        if len(x) != len(y):
            raise Exception("Mismatched Vector Lengths")
        return tuple(i - j for i,j in zip(x,y))

    @staticmethod
    def multiply(x,y):
        if len(x) != len(y):
            raise Exception("Mismatched Vector Lengths")
        return tuple(i * j for i,j in zip(x,y))

    @staticmethod
    def multiply_scalar(s,v):
        return tuple(s * i for i in v)

    @staticmethod
    def dot_prod(x,y):
        if len(x) != len(y):
            raise Exception("Mismatched Vector Lengths")
        return sum(i * j for i,j in zip(x,y))

    @staticmethod
    def length(x):
        return sqrt(Utility.dot_prod(x,x))

    @staticmethod
    def normalize(x):
        l = Utility.length(x)
        return tuple(i / l for i in x)

# material class describes material properties of an object
class Material:
    def __init__(self, color=(255.0,255.0,255.0), specular=10.0, reflectiveness=1.0):
        self.color = color
        self.specular = specular
        self.reflectiveness = reflectiveness

# abstract geometry class
class Geometry(ABC):
    def __init__(self, material=None):
        self.material = material if material else Material()

    @abstractmethod
    def intersections(self, origin, ray):
        return []

    @abstractmethod
    def normal(self, point):
        return (0,0,1)

# parametric sphere
class ParametricSphere(Geometry):
    def __init__(self, center=(0,0,0), radius=1, material=None):
        self.center = center
        self.radius = radius
        super().__init__(material)

    def intersections(self, origin, ray):
        o = Utility.subtract(origin, self.center)
        # parameters for a t^2 + b t + c = 0
        a = Utility.dot_prod(ray, ray)
        b = 2 * Utility.dot_prod(o, ray)
        c = Utility.dot_prod(o, o) - self.radius * self.radius
        # discriminant 
        d = b * b - 4 * a * c
        # no solutions (ray misses sphere)
        if d < 0:
            return []
        # two real solutions (enter/exit sphere)
        # one solution (touched sphere) [t1 == t2]
        t1 = (-b + sqrt(d)) / (2 * a)
        t2 = (-b - sqrt(d)) / (2 * a)
        return [t1, t2]

    def normal(self, point):
        # should I check tht point is actually on geometry?
        return Utility.normalize(Utility.subtract(point, self.center))

# abstract Light class
class Light(ABC):
    def __init__(self, intensities):
        self.intensities = intensities
    
    @abstractmethod
    def intensity(self, ray, point, geometry, scene):
        pass

    def _intensity(self, direction, normal, ray, material):
        i = (0.0, 0.0, 0.0)
        '''
        diffuse lighting
        '''
        t = Utility.dot_prod(normal, direction)
        if t > 0:
            t /= Utility.length(normal) * Utility.length(direction)
            m = Utility.multiply_scalar(t, self.intensities)
            i = Utility.add(i, m)
        '''
        specular lighting
        '''
        if material.specular >= 0:
            r = 2 * Utility.dot_prod(normal, direction)
            r = Utility.multiply_scalar(r, normal)
            r = Utility.subtract(r, direction)
            rv = Utility.dot_prod(r, ray)
            if rv > 0:
                rv = rv / (Utility.length(r) * Utility.length(ray))
                m = Utility.multiply_scalar(rv ** material.specular, self.intensities)
                i = Utility.add(i, m)
        return i

class AmbientLight(Light):
    def __init__(self, intensities):
        super().__init__(intensities)

    def intensity(self, ray, point, geometry, scene):
        return self.intensities

class DirectionalLight(Light):
    def __init__(self, intensities, direction):
        self.direction = direction
        super().__init__(intensities)

    def intensity(self, ray, point, geometry, scene):
        # shadow check
        if scene.intersection(point, self.direction)[1]:
            return (0.0, 0.0, 0.0)
        n = geometry.normal(point)
        m = geometry.material
        return self._intensity(self.direction, n, ray, m)

class PointLight(Light):
    def __init__(self, intensities, position):
        self.position = position
        super().__init__(intensities)

    def intensity(self, ray, point, geometry, scene):
        direction = Utility.subtract(self.position, point)
        # shadow check
        if scene.intersection(point, direction)[1]:
            return (0.0, 0.0, 0.0)
        n = geometry.normal(point)
        m = geometry.material
        return self._intensity(direction, n, ray, m)

# scene class
class Scene:
    def __init__(self, geometry=[], lights=[], background=(0.0,0.0,0.0)):
        self.geometry = geometry
        self.lights = lights
        self.background = background # default color to render 
    
    # return closest geometry that intersects ray
    def intersection(self, origin, ray, tmin=0.001, tmax=float('inf')):
        closest = float('inf')
        geometry = None
        for g in self.geometry:
            for i in g.intersections(origin, ray):
                if i < closest and tmin <= i <= tmax:
                    closest = i
                    geometry = g
        return closest, geometry
    
    def trace_ray(self, origin, ray, tmin=0.001, tmax=float('inf')):
        '''
        Find closest geometry that ray hits
        '''
        closest, geometry = self.intersection(origin, ray, tmin, tmax)
        # did not hit anything
        if not geometry:
            return self.background
        '''
        Compute the color at point on geometry
        ie lighting
        '''
        c = geometry.material.color
        intensity = (0.0, 0.0, 0.0)
        r = Utility.subtract(origin, ray)
        point = Utility.add(origin, Utility.multiply_scalar(closest, ray))
        for l in self.lights:
            intensity = Utility.add(intensity, l.intensity(r, point, geometry, self))
        return Utility.multiply(c, intensity)

# Raytracer class
class RayTracer:
    def __init__(self, viewport=(1.0,1.0,1.0), camera=(0.0,0.0,0.0), rmin=1.0, rmax=float('inf')):
        # should do type checking of parameters here...
        self.viewport = viewport # width, height, depth
        self.camera = camera # x,y,z
        self.rmin = rmin # minimum render distance
        self.rmax = rmax # maximum render distance
    
    # renders given scene to a PIL Image with given dimensions
    def render(self, scene, width=400, height=300):
        # note that this will be [y][x][channel] when access/update
        canvas = np.zeros((height, width, 3))
        for y in range(height):
            for x in range(width):
                vx = (x - width / 2) * self.viewport[0] / width
                vy = (-y + height / 2) * self.viewport[1] / height
                ray = (vx, vy, self.viewport[2])
                canvas[y][x] = scene.trace_ray(self.camera, ray, self.rmin, self.rmax)
        return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

if __name__ == "__main__":
    # add geometry to scene
    g = [
        ParametricSphere((0,-1,3), 1, Material((255,0,0), 500)),
        ParametricSphere((2,0,4), 1, Material((0,0,255), 500)),
        ParametricSphere((-2,0,4), 1, Material((0,255,0), 10)),
        ParametricSphere((0,-5001,0), 5000, Material((255,255,0), 1000)), # yellow
    ]
    # add lighting to scene
    l = [
        AmbientLight((0.2,0.2,0.2)),
        DirectionalLight((0.2,0.2,0.2), (1,4,4)),
        PointLight((0.6,0.6,0.6), (2,1,0)),
    ]
    # render scene
    RayTracer().render(Scene(g,l), 300, 300).save("Renders/canvas.png")
