"""
Created on Fri May 10 14:58:31 2019

@author: zemmouri
"""

import math

class Point :
    """ Point : a class reprensenting points in IR3 """
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
    
    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else :
            return False
      
    def distance_origine(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def distance (self, p):
        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2 + (self.z - p.z)**2)


class ColoredPoint (Point):
    """ ColoredPoint : a class reprensenting colored points in IR3 """
    def __init__(self, x=0, y=0, z=0, color='black'):
        super().__init__(x, y, z)
        self.color = color
    
    def __str__(self):
        return "(Colored " + self.color + ", " + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"


class Forme:
    def __init__(self, c, d):
        self.c = c  # centre de la forme
        self.d = d  # densité de la forme (pour le poids)
    
    def __str__(self):
        return self.__class__.__name__ + " : centre = " + self.c.__str__() + " , densite = " + str(self.d)  

    def surface(self):
        pass
    def volume (self):
        pass
    def poids(self):
        return self.d * self.volume()



class Sphere (Forme):
    def __init__(self, c, r, d):
        super().__init__(c, d)
        self.r = r
    
    def __str__(self):
        return super().__str__() + " , rayon = " + str(self.r)

    def surface (self):
        return math.pi * self.r * self.r * 4
        
    def volume (self):
        return math.pi * (self.r ** 3) * 4 / 3

class Cube (Forme) :
    def __init__(self, c, a, d):
        super().__init__(c, d)
        self.a = a

    def __str__(self):
        return super().__str__() + " , cote = " + str(self.a)

    def surface (self):
        return 6 * self.a * self.a
    
    def volume (self):
        return self.a ** 3

class Cylindre (Forme) :
    def __init__(self, c, r, h, d):
        super().__init__(c, d)
        self.r = r
        self.h = h

    def __str__(self):
        return super().__str__() + " , rayon = " + str(self.r) + " , hauteur = " + str(self.h)

    def surface (self):
        return 2 * math.pi * self.r * self.h
    
    def volume (self):
        return math.pi * self.r * self.r * self.h





a = Point(1, 1, 1)
print(a)
b = ColoredPoint(10, 11, 10, 'red')
print(b)

isinstance(b, ColoredPoint)
isinstance(b, Point)

f1 = Forme(Point(1, 2, 3), 1.5)
print(f1.c)
print(f1.d)
print(f1)
print(f1.surface())
print(f1.volume())


f2 = Sphere(a, 2, 12.5)
print(f2)
print(f2.surface())
print(f2.volume())
print(f2.poids())

isinstance(f2, Sphere)
isinstance(f2, Forme)

f3 = Cube(b, 3, 1.5)
print(f3)
print(f3.surface())
print(f3.volume())
print(f3.poids())

f4 = Cylindre(b, 1, 10, 3.5)
print(f4)
print(f4.surface())
print(f4.volume())
print(f4.poids())

L = [f2, f3, f4]

for f in L:
    print(f)
    print("surface = ", f.surface())
    print("volume  = ", f.volume())
    print("poids   = ", f.poids())
    print()

