using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetNums
{
    public class Matrix
    {
        private double[,] matrix;
        private static MatrixHelper matrixHelper = new MatrixHelper();

        private static Random rnd = new Random();


        public static Matrix Ones(int height, int width)
        {
            Matrix a = new Matrix(height, width);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    a[i, j] = 1;
                }
            }

            return a;
        }

        public static Matrix Randoms(int height, int width)
        {
            Matrix a = new Matrix(height, width);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    a[i, j] = rnd.NextDouble();
                }
            }

            return a;
        }

        public Matrix(double[,] matrix)
        {
            this.matrix = matrix;
        }

        public Matrix(int i, int j)
        {
            matrix = new double[i, j];
        }

        public Matrix(Matrix matrix)
        {
            //this.matrix = matrix.GetMatrix();
            this.matrix = new double[matrix.GetLength(0), matrix.GetLength(1)];
            for(int i=0;i<matrix.GetLength(0);i++)
            {
                for(int j=0;j<matrix.GetLength(1);j++)
                {
                    this.matrix[i, j] = matrix[i, j];
                }
            }
        }

        public Matrix(double[] vector)
        {
            this.matrix = new double[vector.Length, 1];
            for (int i = 0; i < vector.Length; i++)
            {
                matrix[i, 0] = vector[i];
            }
        }


        public Matrix ElemntWiseMult(Matrix a)
        {
            Matrix aux = new Matrix(matrix.GetLength(0), matrix.GetLength(1));

            for(int i=0;i<aux.GetLength(0);i++)
            {
                for(int j=0;j<aux.GetLength(1);j++)
                {
                    aux[i, j] = matrix[i, j] * a[i, j];
                }
            }

            return aux;
        }

        public Matrix GetSqareByElement()
        {
            return new Matrix(matrixHelper.SquaredByElement(matrix));
        }

        public Matrix GetTranspose()
        {
            return matrixHelper.GetTranspose(matrix);
        }

        public int GetLength(int i)
        {
            return matrix.GetLength(i);
        }

        public double[,] GetMatrix()
        {
            return matrix;
        }

        public double this[int i, int j]
        {
            get { return matrix[i, j]; }
            set { matrix[i, j] = value; }
        }


        public override string ToString()
        {
            string mToSt = "";
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    mToSt += matrix[i, j] + " ";
                }
                mToSt += "\n";
            }

            return mToSt;
        }

        public double GetSum()
        {
            double b = 0;
            for(int i=0;i<matrix.GetLength(0);i++)
            {
                for(int j=0;j<matrix.GetLength(1);j++)
                {
                    b += matrix[i, j];
                }
            }

            return b = b / matrix.Length;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            return matrixHelper.Addition(a, b);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            return matrixHelper.Substraction(a, b);
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            return matrixHelper.MatrixMultiplication(a, b);
        }

        public static Matrix operator *(Matrix a, double b)
        {
            return matrixHelper.Multiplication(a, b);
        }

        public static Matrix operator *(double b, Matrix a)
        {
            return matrixHelper.Multiplication(a, b);
        }

        public static Matrix operator /(Matrix a, double b)
        {
            return matrixHelper.Division(a, b);
        }

        public static Matrix operator *(Matrix a, double[] b)
        {
            return matrixHelper.ElementWiseMult(a, b);
        }
    }
}
