using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetNums
{
    public class MatrixHelper
    {
        private static Random rnd;

        public MatrixHelper()
        {
            rnd = new Random();
        }

        public Matrix MatrixMultiplication(Matrix a, Matrix b)
        {
            Matrix multipliedMatrix = new Matrix(a.GetLength(0), b.GetLength(1));

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < b.GetLength(1); j++)
                {
                    double value = 0;
                    for (int q = 0; q < a.GetLength(1); q++)
                    {
                        value += a[i, q] * b[q, j];
                    }
                    multipliedMatrix[i, j] = value;
                }
            }

            return multipliedMatrix;
        }


        public Matrix ElementWiseMult(Matrix a, double[] b)
        {
            Matrix multMatrix = new Matrix(a.GetLength(0), a.GetLength(1));
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    multMatrix[i, j] = a[i, j] * b[i];
                }
            }
            return multMatrix;
        }

        public Matrix Addition(Matrix a, Matrix b)
        {
            Matrix addedMatrix = new Matrix(a.GetLength(0), a.GetLength(1));

            for (int i = 0; i < addedMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < addedMatrix.GetLength(1); j++)
                {
                    addedMatrix[i, j] = a[i, j] + b[i, j];
                }
            }

            return addedMatrix;
        }

        public Matrix Substraction(Matrix a, Matrix b)
        {
            Matrix substractadMatrix = new Matrix(a.GetLength(0), a.GetLength(1));

            for (int i = 0; i < substractadMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < substractadMatrix.GetLength(1); j++)
                {
                    substractadMatrix[i, j] = a[i, j] - b[i, j];
                }
            }

            return substractadMatrix;
        }

        public Matrix Multiplication(Matrix a, double b)
        {
            Matrix matrix = new Matrix(a.GetLength(0), a.GetLength(1));

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    matrix[i, j] = a[i, j] * b;
                }
            }

            return matrix;
        }

        public Matrix Division(Matrix a, double b)
        {
            Matrix matrix = new Matrix(a.GetLength(0), a.GetLength(1));

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    matrix[i, j] = a[i, j] / b;
                }
            }

            return matrix;
        }

        public Matrix GetTranspose(double[,] matrix)
        {
            Matrix transposedMatrix = new Matrix(matrix.GetLength(1), matrix.GetLength(0));

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    transposedMatrix[j, i] = matrix[i, j];
                }
            }

            return transposedMatrix;
        }

        public Matrix SquaredByElement(double[,] a)
        {
            Matrix b = new Matrix(a);
            for (int i = 0; i < b.GetLength(0); i++)
            {
                for (int j = 0; j < b.GetLength(1); j++)
                {
                    b[i, j] *= b[i, j];
                }
            }

            return b;
        }
    }
}
