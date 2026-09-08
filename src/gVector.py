"""
gVector: A simple vector class for basic linear algebra operations.

The gVector class represents a mathematical vector as a list of numeric
values and provides support for vector arithmetic, scalar multiplication,
dot products, vector norms, and vector projection. It also implements
common Python operators for indexing, assignment, comparison, addition,
subtraction, and scalar multiplication.

Supported operations include:
- Vector addition and subtraction
- Scalar multiplication
- Dot product
- Euclidean (L2), Manhattan (L1), and infinity (L-infinity) norms
- Projection of one vector onto another
- Element access and assignment
- Vector dimension and value retrieval

Vectors are required to be represented internally as Python lists.
"""
#
# To Do:
#
#     1. Add type checking for other all places where used.
#     2. Update comments everywhere.
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
#
import math

class gVector(object):
    """
    A vector class supporting linear algebra operations.

    Attributes:
        values (list): A list of numeric values representing the vector.
    """

    def __init__(self, values):
        """
        Initialize a vector with a list of values.

        Args:
            values (list): List of numeric values.

        Raises:
            ValueError: If values is not a list.
        """
        if not isinstance(values, list):
            raise ValueError('Vector values must be of type list. Use [] for empty vector.')
        self.values = values

    def __len__(self):
        """
        Return the dimension (length) of the vector.
        """
        return len(self.values)

    def __eq__(self, other):
        """
        Check if two vectors are equal.

        Args:
            other (gVector): Another vector.

        Returns:
            bool: True if values are equal, False otherwise.
        """
        return self.values == other.values

    def __ne__(self, other):
        """
        Check if two vectors are not equal.
        """
        return not self.__eq__(other)

    def __getitem__(self, index):
        """
        Get the value at a specific index.

        Args:
            index (int): Index of element.

        Returns:
            numeric: Value at index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index >= len(self):
            raise IndexError('Index out of range.')
        return self.values[index]

    def __setitem__(self, index, value):
        """
        Set the value at a specific index.

        Args:
            index (int): Index of element.
            value (numeric): Value to assign.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index >= len(self):
            raise IndexError('Index out of range.')
        self.values[index] = value

    def __str__(self):
        """
        Return string representation of the vector.
        """
        return f"{self.values}"

    def __add__(self, other):
        """
        Add two vectors element-wise.

        Args:
            other (gVector): Vector to add.

        Returns:
            gVector: Resulting vector.

        Raises:
            ValueError: If dimensions do not match.
        """
        if len(self) != len(other):
            raise ValueError('Vectors must have equal dimensions to add.')
        return gVector([a + b for a, b in zip(self.values, other.values)])

    def __sub__(self, other):
        """
        Subtract two vectors element-wise.

        Args:
            other (gVector): Vector to subtract.

        Returns:
            gVector: Resulting vector.

        Raises:
            ValueError: If dimensions do not match.
        """
        if len(self) != len(other):
            raise ValueError('Vectors must have equal dimensions to subtract.')
        return gVector([a - b for a, b in zip(self.values, other.values)])

    def __mul__(self, value):
       """
       Multiply vector by a scalar.

       Args:
           value (int or float): Scalar multiplier (A * 3).

       Returns:
           gVector: New scaled vector.

       Raises:
           ValueError: If value is not numeric.
       """
       if not isinstance(value, (int, float)):
           raise ValueError('Multiplier must be int or float.')
       return gVector([x * value for x in self.values])

    def __rmul__(self, value):
      """
      Reflected multiply vector by a scalar (3 * A).

      Args:
          value (int or float): Scalar multiplier.

      Returns:
          gVector: New scaled vector.

      Raises:
          ValueError: If value is not numeric.
      """
      return self.__mul__(value)

    def __truediv__(self, value):
        """
        Divide vector by a scalar.

        Args:
            value (int or float): Scalar divisor (A / 3).

        Returns:
            gVector: New scaled vector.

        Raises:
            ValueError: If value is not numeric.
        """
        if not isinstance(value, (int, float)):
            raise ValueError('Divisor must be int or float.')
        if value == 0:
            raise ZeroDivisionError('Attempted to divide by zero.')
        return gVector([x / value for x in self.values])

    def get_dim(self):
        """
        Get the dimension of the vector.

        Returns:
            int: Number of elements.
        """
        return len(self)

    def set_values(self, values):
        """
        Replace vector values.

        Args:
            values (list): New values.

        Raises:
            ValueError: If values is not a list.
        """
        if not isinstance(values, list):
            raise ValueError('Vector values must be of type list.')
        self.values = values

    def get_values(self):
        """
        Get the underlying list of values.

        Returns:
            list: Vector values.
        """
        return self.values

    def dot(self, other):
        """
        Compute dot product with another vector.

        Args:
            other (gVector): Other vector.

        Returns:
            numeric: Dot product.

        Raises:
            ValueError: If dimensions do not match.
        """
        if len(self) != len(other):
            raise ValueError('Vectors must have equal dimensions.')
        return sum(a * b for a, b in zip(self.values, other.values))

    def cross(self, other):
        """
        Compute the Cross Product of two 3D Vectors

        Args:
            other (gVector): Other vector.

        Returns:
            gVector: 3D vector cross product.

        Raises:
            ValueError: If both vectors are not 3D.
        """
        if len(self) != 3 or len(other) != 3:
            raise ValueError('Cross products can only be computed for two 3D vectors.')
        return gVector([ (self[1]*other[2] - self[2]*other[1]),
                         (self[2]*other[0] - self[0]*other[2]),
                         (self[0]*other[1] - self[1]*other[0])])

    def norm(self):
        """
        Compute the Euclidean (L2) norm.

        Returns:
            float: sqrt(sum(x^2))
        """
        return math.sqrt(sum(x**2 for x in self.values))

    def manhattan_norm(self):
        """
        Compute the Manhattan (L1) norm.

        Returns:
            float: sum(|x|)
        """
        return sum(abs(x) for x in self.values)

    def infinity_norm(self):
        """
        Compute the infinity norm.

        Returns:
            float: max(|x|)
        """
        return max(abs(x) for x in self.values)

    def projection_onto(self, other):
        """
        Project this vector onto another vector.

        Formula:
            proj_B(A) = (A · B / ||B||^2) * B

        Args:
            other (gVector): Vector to project onto.

        Returns:
            gVector: Projection vector.
        """
        scale = self.dot(other) / (other.norm() ** 2)
        return other * scale

    def normalize(self):
        """
        Generate a Unit vector in the same direction of this vector.

        Returns:
            gVector: a new unit vector.
        """
        return self / self.norm()
    
    def is_zero(self):
        """
        Test if this vector is the Zero vector.

        Returns:
            bool: True if Zero vector
        """
        return all(x == 0 for x in self.values)
    
    def is_orthogonal(self, other):
        """
        Test if this vector is Orthogonal to the other vector.

        Returns:
            bool: True if the vectors are orthogonal.

        Raises:
            ValueError: If dimensions do not match.
        """
        if len(self) != len(other):
            raise ValueError('Vectors must have equal dimensions to check orthogonality.')
        return (self.dot(other) == 0)
    
    def angle(self, other):
        """
        Calculate the angle between two vectors.

        Returns:
            float: The angle between this and other in radians.

        Raises:
            ValueError: If dimensions do not match.

        """
        if len(self) != len(other):
            raise ValueError('Vectors must have equal dimensions to determine angle.')
        return(math.acos( self.dot(other) / (self.norm() * other.norm()) ))
      
    def distance(self, other):
        """
        Calculate the Euclidean distance between two vectors.
        
        The distance is defined as the Euclidean norm of the difference
        between the two vectors:
            distance(A, B) = ||A - B||
            
        Args: other (gVector): The vector to measure the distance to.
        
        Returns: float: The Euclidean distance between the two vectors.
        
        Raises: TypeError: If other is not a gVector.
                ValueError: If the vectors have different dimensions.
                
        """
        if not isinstance(other, gVector):
            raise TypeError('Distance requires another vector.')

        if len(self) != len(other):
            raise ValueError('Vectors must have equal dimensions.')

        return( (self - other).norm() )
