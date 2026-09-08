import math
import gVector

 
class gMatrix(object):
    def __init__(self, values): 
        # Check that the values are a list of lists.
        if not all(isinstance(v, list) for v in values):
            raise TypeError('Matrix values are of type list of lists. Pass [[]] if no values.')

        # Store the values and row count.
        self.values = values
        self.m = len(values)
        
        # Store the column count. Avoid a zero row column index error.
        if self.m > 0: self.n = len(values[0]) 
        else: self.n = 0
        
        # Check that all rows have the same number of columns.
        for r in values:
            if not len(r) == self.n:
                raise ValueError('All rows of the matrix must have the same number of columns.')

    @classmethod
    def zero(cls, nrows, ncols):
        if not nrows >= 0 and ncols >= 0:
            raise ValueError('The all zero matrix must have non-negative dimensions.')
        return cls([[0 for col in range(ncols)] for row in range(nrows)] )

    def __len__(self):
        return len(self.values)
    
    def __eq__(self, other):
        return self.values == other.values
    
    def __ne__(self, other):
        return not self.values == other.values
    
    def __getitem__(self, index):
        row, column = index
        if not (0 <= row < self.m and 0 <= column < self.n):
            raise IndexError('Indecies out of range.')
        return(self.values[row][column])

    def __setitem__(self, index, value):
        row, column = index
        if not (0 <= row < self.m and 0 <= column < self.n):
            raise IndexError('Indecies out of range.')
        self.values[row][column] = value

    def __str__(self):
        # Find the widest formatted value in the matrix and use it as the
        # column width.
        cw=max(max(len(str(v)) for v in r) for r in self.values)
        # Format each row with space-delimited values formatted to the width.
        return '\n'.join(' '.join(f'{v:{cw}}' for v in r) for r in self.values)

    def __add__(self, other):
        if not isinstance(other, gMatrix):
            raise TypeError('Addition must be of two matrices.')
        if self.m != other.m or self.n != other.n:
            raise ValueError('Matrices must have equal dimensions to add.')
        return gMatrix([[a+b for a, b in zip(srow, orow)] for srow, orow in zip(self.values, other.values)])

    def __sub__(self, other):
        if not isinstance(other, gMatrix):
            raise TypeError('Subtraction must be of two matrices.')
        if self.m != other.m or self.n != other.n:
            raise ValueError('Matrices must have equal dimensions to subtract.')
        return gMatrix([[a-b for a, b in zip(srow, orow)] for srow, orow in zip(self.values, other.values)])

    def __mul__(self, other):
        # Supports an element by element matrix multiplication times a scaler
        # or another matrix of the same size. This is also known as the
        # Hadamard product. This is for the operation A*B.
        #
        # Check if the argument is a scaler and if so, perform a scaler
        # multiplication.
        if isinstance(other, (int, float)):
            return gMatrix([[a*other for a in row] for row in self.values])
        
        # Otherwise, check if it is another matrix of the same dimension, then
        # compute the Hadamard product.
        elif isinstance(other, gMatrix):
            if self.m != other.m or self.n != other.n:
                raise ValueError('Matrices must have equal dimensions to multiply.')
            return gMatrix([[a*b for a, b in zip(srow, orow)] for srow, orow in zip(self.values, other.values)])
        else:
            raise TypeError('Multiplication must be by a scaler or another matrices.')
            
    def __matmul__(self, other):
        """Perform matrix multiplication of self and other.

        For matrices A and B, the operation A @ B is valid when the
        number of columns in A equals the number of rows in B. The
        resulting matrix has the number of rows of A and the number
        of columns of B.

        Args:
            other (gMatrix): The right-hand matrix in the multiplication.

        Returns:
            gMatrix: The matrix product self @ other.

        Raises:
            TypeError: If other is not a gMatrix.
            ValueError: If the matrices have incompatible dimensions.
        """
        if not isinstance(other, gMatrix):
            raise TypeError('Matrix multiplication requires another matrix.')

        if self.ncols() != other.nrows():
            raise ValueError(
                'Matrix A column count must equal matrix B row count.'
                )

        # Transpose B so that its columns become rows. This allows each
        # element of C to be calculated as the dot product of a row of A
        # with a row of B transpose.
        # Note: I could save repeated transposes by storing the results
        # on the first iteration and then looking them up on subsequesnt
        # iteraations.
        C = gMatrix.zero(self.nrows(), other.ncols())
        Bt = other.T()

        for r, Arow in enumerate(self.values):
            Av = gVector.gVector(Arow)
            for c, Bcol in enumerate(Bt.values):
                C[r, c] = Av.dot(gVector.gVector(Bcol))

        return C

    def nrows(self):
        return(len(self))
    
    def ncols(self):
        return len(self.values[0]) if self.nrows() > 0 else 0

    def max(self):
        return max(map(max,self.values))

    def min(self):
        return min(map(min,self.values))

    def T(self):
        return gMatrix([[self[col,row] for col in range(self.nrows())] for row in range(self.ncols())])

if __name__ == "__main__":
    m = gMatrix([[1,2,300],[4,5,6.7],[-600,7.999999,8]])
    
    print ("0,1=", m.values[0][1])
    print( m )

    a = gMatrix([[1,2,3],[4,5.4,6],[7,8,9]])
    b = gMatrix([[0,1,2],[10,11,12],[13,14,15]])
    # x = a + b
    print( a )
    # c = jMatrix([[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]])
    # print( a + c )
    
    t = a.T()
    print( t )
    
    print("Scaler Hadamard product:", a*5 )
    print("Matrix Hadamard product:", a*b )
    print("Matrix mult:", a@b )
