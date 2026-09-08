# -*- coding: utf-8 -*-
#
# Copyright (C) James Garvey
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3. This program is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see https://www.gnu.org/licenses/.
import gVector

#if __name__ == "__main__":
x = gVector.gVector([1,2,3,4,5])
y = gVector.gVector([6,7,8,9,10])
        
print( x )
print("Length of x:", len(x) )
print( "Value of x[2]:", x[2] ) 
print( y )
print("Length of y:", len(y) )
print( "Value of y[2]:", y[2] )

v = gVector.gVector([1,1])
q = v / 5
print("v div by 5:", q)

u = v.normalize()
print("v normalized:", u)
    
z = x + y
print( "Sum x+y:", z )
    
d = x.dot(y)
print( "Dot x.y:", d )
    
print( "Scaler product y*5:", y * 5)
print( "Scaler rproduct 5*y:", 5 * y)
    
print( "Euclidean norm x:", x.norm())
print( "Manhatten norm x:", x.manhattan_norm())
print( "Infinity norm x: ", x.infinity_norm())
    
a = gVector.gVector([13,3])
b = gVector.gVector([3,10])

print( "A =", a )
print( "B =", b )
print( "A dot B", a.dot(b))
print( "B length and length squared", b.norm(), b.norm()**2)
    
print( a.projection_onto(b))

z = gVector.gVector([0,0,0,0,0])
print( "z is the zero vector:", z.is_zero())
print( "a is not the zero vector:", a.is_zero())

print ("Are x and y orthogonal:", x.is_orthogonal(y))
print ("Are x and z orthogonal:", x.is_orthogonal(z))

d = gVector.gVector([1,1])
ux = gVector.gVector(([1,0]))

print( "Angle of d:", d.angle(ux))

crossx = gVector.gVector([1,0,0])
crossy = gVector.gVector([0,-1,0])
crossz = crossx.cross(crossy)

print( "cross product is:", crossz )

print( "distance between cross x, y:", crossx.distance(crossy))

# Error case testing:
#print( "Illegal cross product (5x3):", x.cross(crossx))
#print( "Value of x[10]:", x[10] )
