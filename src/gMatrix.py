import math
import gVector
import enum

TOL = 1e-12

# ToDo:
# - add slicing to getitem
#

class GSolveStatus(enum.Enum):
    SOLVED = 'solved'
    NO_SOLUTION = 'no solution'
    INFINITE_SOLUTIONS = 'infinite solutions'
 
class GMatrix(object):
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
        # If index is a tuple, then use 2d indexing.
        if isinstance(index, tuple):
            row, column = index
            if not (0 <= row < self.m and 0 <= column < self.n):
                raise IndexError('Indicies out of range.')
            return(self.values[row][column])
        # Otherwise, return the row value list.
        elif isinstance(index, int):
            if not (0 <= index < self.m):
                raise IndexError('Indicies out of range.')
            return(self.values[index])
        # Unsupported index type.
        else:
            raise TypeError('Index must be a tuple (row,column) or an int (row).')       

    def __setitem__(self, index, value):
        row, column = index
        if not (0 <= row < self.m and 0 <= column < self.n):
            raise IndexError('Indecies out of range.')
        self.values[row][column] = value
    
    def __str__(self):
        # Find the widest formatted value in the matrix.
        cw = max(8, max(len(f'{v:.8g}') for row in self.values for v in row)) + 1

        # Format each row with right-aligned, fixed-width values.
        return '\n'.join(
            ''.join(f'{v:>{cw}.8g}' for v in row)
                for row in self.values
                )

    def __add__(self, other):
        if not isinstance(other, GMatrix):
            raise TypeError('Addition must be of two matrices.')
        if self.m != other.m or self.n != other.n:
            raise ValueError('Matrices must have equal dimensions to add.')
        return GMatrix([[a+b for a, b in zip(srow, orow)] for srow, orow in zip(self.values, other.values)])

    def __sub__(self, other):
        if not isinstance(other, GMatrix):
            raise TypeError('Subtraction must be of two matrices.')
        if self.m != other.m or self.n != other.n:
            raise ValueError('Matrices must have equal dimensions to subtract.')
        return GMatrix([[a-b for a, b in zip(srow, orow)] for srow, orow in zip(self.values, other.values)])

    def __mul__(self, other):
        # Supports an element by element matrix multiplication times a scaler
        # or another matrix of the same size. This is also known as the
        # Hadamard product. This is for the operation A*B.
        #
        # Check if the argument is a scaler and if so, perform a scaler
        # multiplication.
        if isinstance(other, (int, float)):
            return GMatrix([[a*other for a in row] for row in self.values])
        
        # Otherwise, check if it is another matrix of the same dimension, then
        # compute the Hadamard product.
        elif isinstance(other, GMatrix):
            if self.m != other.m or self.n != other.n:
                raise ValueError('Matrices must have equal dimensions to multiply.')
            return GMatrix([[a*b for a, b in zip(srow, orow)] for srow, orow in zip(self.values, other.values)])
        else:
            raise TypeError('Multiplication must be by a scaler or another matrices.')
            
    def __matmul__(self, other):
        """Perform matrix multiplication of self and other.

        For matrices A and B, the operation A @ B is valid when the
        number of columns in A equals the number of rows in B. The
        resulting matrix has the number of rows of A and the number
        of columns of B.

        Args:
            other (GMatrix): The right-hand matrix in the multiplication.

        Returns:
            GMatrix: The matrix product self @ other.

        Raises:
            TypeError: If other is not a GMatrix.
            ValueError: If the matrices have incompatible dimensions.
        """
        if not isinstance(other, GMatrix):
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
        # iterations.
        C = GMatrix.zero(self.nrows(), other.ncols())
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
        return GMatrix([[self[col,row] for col in range(self.nrows())] for row in range(self.ncols())])
    
    def copy(self):
        return GMatrix([row.copy() for row in self.values])
    
    def scale_row(self, row, factor):
        # Scale every element in the specified row by factor.
        if not isinstance(row, int):
            raise TypeError('row index must be an integer.')

        if not isinstance(factor, (int, float)):
            raise TypeError('scale factor must be an integer or float.')

        if not 0 <= row < len(self):
            raise IndexError(f'row index {row} is outside the matrix bounds.')

        self.values[row] = [
            0.0 if abs(v := a * factor) < TOL else v for a in self.values[row]
            ]

    def swap_rows(self, arow, brow):
        if not isinstance(arow, int) or not isinstance(brow, int):
            raise TypeError('row indicies must be an integer.')
            
        if not 0 <= arow < len(self):
            raise IndexError(f'arow index {arow} is outside the matrix bounds.')
            
        if not 0 <= brow < len(self):
            raise IndexError(f'brow index {brow} is outside the matrix bounds.')
            
        self[arow], self[brow] = self[brow], self[arow]

    def add_row_mult(self, arow, brow, factor):
        if not isinstance(arow, int) or not isinstance(brow, int):
            raise TypeError('row indicies must be integers.')

        if not isinstance(factor, (int,float)):
            raise TypeError('multiplier factor must be integer or float.')
            
        if not 0 <= arow < len(self):
            raise IndexError(f'arow index {arow} is outside the matrix bounds.')
            
        if not 0 <= brow < len(self):
            raise IndexError(f'brow index {brow} is outside the matrix bounds.')
        
        self.values[arow] = [
            0.0 if abs(v := a + b * factor) < TOL else v
                for a, b in zip(self.values[arow], self.values[brow])
                ]

    def ref(self):
        rmat = self.copy()
        nrows = rmat.nrows()
        ncols = rmat.ncols()
        
        prow = 0
        for pcol in range(ncols):
            
            # Find a non-zero entry in this column to use as the pivot.
            for srow in range(prow, nrows):
                if abs(rmat[(srow,pcol)]) > TOL:
                    # Pivot found.
                    
                    # If the search row is not the pivot row, then swap
                    # the current row with the pivot row.
                    if prow != srow: 
                        rmat.swap_rows(prow, srow)
                    
                    # Use the pivot row to zero all of the values in the pivot
                    # column below the current row.
                    for zrow in range(prow+1, nrows):
                        factor = -rmat[(zrow, pcol)] / rmat[(prow, pcol)]
                        rmat.add_row_mult(zrow, prow, factor)
                    
                    # Move on to the next pivot row.
                    prow += 1
                    break
            
            # If the pivot row was the last row, stop the matrix is complete.
            if prow >= rmat.nrows():
                break
        
        return rmat
    
    def rref(self):
        rrmat = self.ref()
        # Initialize the pivot position lookup dictionary.
        piv_cols={}

        nrows = rrmat.nrows()
        ncols = rrmat.ncols()
        
        # Normalize the pivots.
        for row in range(nrows):
            for col in range(ncols):
                pval = rrmat[(row,col)]
                if abs(pval) > TOL:
                    piv_cols[row] = col
                    rrmat.scale_row( row, 1/pval )
                    break
        
        # Start on the last row of the matrix and use the
        # pivot value to zeroize the pivot column of all
        # rows above.
        for row in range(nrows-1, -1, -1):
            pcol = piv_cols.get(row)
                        
            if pcol is not None:
                for zrow in range(row):
                    factor = -rrmat[(zrow,pcol)]
                    rrmat.add_row_mult(zrow, row, factor)
        
        return rrmat
    
    def augment(self, b):
        # Check that b is a list of numeric and that its length
        # is equal to the matrix row count.
        if not isinstance(b, list):
            raise TypeError('RHS values must be a list of numeric values.')
        if len(b) != self.nrows():
            raise ValueError('RHS dimension must equal matrix row count.')
        if not all(isinstance(bv, (int, float)) for bv in b):
            raise TypeError('RHS values must be numeric values.')
            
        amat = self.copy()
        for r, row in enumerate(amat):
            row.append(b[r])
        return amat
    
    def solve( self, b ):
        # Check that b is a list of numerics and that its length is
        # equal to the row count.
        if not isinstance(b, list):
            raise TypeError('RHS values must be a list of numeric values.')
        if len(b) != self.nrows():
            raise ValueError('RHS dimension must equal matrix row count.')
        if not all(isinstance(bv, (int, float)) for bv in b):
            raise TypeError('RHS values must be numeric values.')
            
        #Initialize the result.
        solution = None
        status = GSolveStatus.SOLVED

        # Create the augmented matrix.
        amat = self.augment(b)
        
        # Convert it to Reduced Row Echelon form.
        rrmat = amat.rref()
        
        # Check for consistancy and count pivot rows.
        pivot_count = 0
        for row in rrmat: 
            # Check for the row condition of all 0 coefficients.
            if all( abs( rv ) <= TOL for rv in row[:-1]): 
                # If the RHS is non-zero, then the equations are inconsistant.
                if abs( row[-1] ) > TOL:
                    status = GSolveStatus.NO_SOLUTION
                    break
            else:
                pivot_count += 1
        
        # If the matrix is consistent and the pivot count is less than
        # the column count, then there are free variables and an infinite
        # number of soltions. Otherwise the RHS column contains the solution.
        if status == GSolveStatus.SOLVED:
            if( pivot_count < self.ncols() ):
                status = GSolveStatus.INFINITE_SOLUTIONS
            else:
                rhs = self.ncols()
                solution = []
                for row in range(rrmat.ncols()-1):
                    solution.append(rrmat[row][rhs])
        
        return (solution, status)
     
if __name__ == "__main__":
    m = GMatrix([[1,2,300],[4,5,6.7],[-600,7.999999,8]])
    
    print ("0,1=", m.values[0][1])
    print( m )

    a = GMatrix([[1,2,3],[4,5.4,6],[7,8,9]])
    b = GMatrix([[0,1,2],[10,11,12],[13,14,15]])
    # x = a + b
    print( a )
    
    c = GMatrix([[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]])
    print( a + c )
    
    print( "Unscaled:", c )
    c.scale_row(1 , 5)
    print( "Scaled:", c )
    
    t = a.T()
    print( t )
    
    print("Scaler Hadamard product:", a*5 )
    print("Matrix Hadamard product:", a*b )
    print("Matrix mult:", a@b )
    
    print( a )
    print( a.ref() )
    print( a.rref() )
    
    s = GMatrix([ [2,1,-1], [1,3,2], [3,-1,1] ])
    b = [4,7,5]
    x = s.augment(b)
    
    print("original:\n", x)
    print("ref :\n", x.ref())
    print("rref:\n", x.rref())

    print( s.solve(b) )
    
    s = GMatrix([ [2,1], [1,-1], [3,2] ])
    b = [5,1,8]
    x = s.augment(b)
    
    print("original:\n", x)
    print("ref :\n", x.ref())
    print("rref:\n", x.rref())
    
    print( s.solve(b) )

    s = GMatrix([
        [1, 2, 1],
        [2, 4, 3]
        ])

    b = [4, 9]
    
    x = s.augment(b)
    
    print("original:\n", x)
    print("ref :\n", x.ref())
    print("rref:\n", x.rref())
    
    print( s.solve(b) )
   
    
    
