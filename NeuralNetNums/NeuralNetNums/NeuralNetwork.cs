using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetNums
{
    public class NeuralNetwork
    {
        public RichTextBox rtb;

        public static Random rnd = new Random();

        public double learningRate = 0.5;

        public List<Layer> layers;

        public Matrix inputs;
        public Matrix expectedOutputs;

        public Matrix expectedALL;

        public Matrix Cost;

        public Matrix[] errors;
        public Matrix[] deltas;

        public NeuralNetwork(double[,] inputs, double[,] expectedOutputs, int[] layerCounts)
        {
            errors = new Matrix[layerCounts.Length - 1];
            deltas = new Matrix[layerCounts.Length - 1];

            layers = new List<Layer>();
            this.inputs = new Matrix(inputs);
            this.expectedOutputs = new Matrix(expectedOutputs);

            layers.Add(new Layer());
            layers[0].neuronCount = layerCounts[0];

            layers[0].w = Matrix.Randoms(layerCounts[0], layerCounts[1]);

            layers[0].b = Matrix.Ones(1, layerCounts[0]);

            layers[0].a = new Matrix(inputs);

            for (int i = 1; i < layerCounts.Length - 1; i++)
            {
                layers.Add(new Layer());
                layers[i].neuronCount = layerCounts[i];

                layers[i].w = Matrix.Randoms(layerCounts[i], layerCounts[i + 1]);

                layers[i].b = Matrix.Ones(1, layerCounts[i]);
            }

            layers.Add(new Layer());
            layers[layers.Count - 1].neuronCount = layerCounts[layerCounts.Length - 1];

            layers[layers.Count - 1].b = Matrix.Ones(1, layerCounts[layers.Count - 1]);
        }


        public void TrainAllA()
        {
            RunWithoutTrain();

            layers[layers.Count - 1].error = (layers[layers.Count - 1].a - expectedOutputs);

            Cost = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(layers[layers.Count - 1].a - expectedOutputs);

            for (int i = layers.Count - 2; i > 0; i--)
            {
                layers[i].error = (SigmoidDerAct(layers[i + 1].a).ElemntWiseMult(layers[i + 1].error)) * layers[i].w.GetTranspose();
            }

            for (int i = 0; i < layers.Count - 1; i++)
            {
                layers[i].delta = layers[i].a.GetTranspose() * (SigmoidDerAct(layers[i + 1].a).ElemntWiseMult(layers[i + 1].error));
            }

            for (int i = 0; i < layers.Count - 1; i++)
            {
                layers[i].w = layers[i].w - ((layers[i].delta / layers[0].a.GetLength(0)) * learningRate);
            }

            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].b = layers[i].b - ((layers[i].error.ElemntWiseMult(SigmoidDerAct(layers[i].a))).AvgPerRow() * learningRate);
            }
        }


        public void TrainAllZ()
        {
            RunWithoutTrain();
            layers[layers.Count - 1].error = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(SigmoidDerAct(layers[layers.Count - 1].a));

            Cost = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(layers[layers.Count - 1].a - expectedOutputs);

            for (int i = layers.Count - 2; i > 0; i--)
            {
                layers[i].error = (layers[i + 1].error * layers[i].w.GetTranspose()).ElemntWiseMult(SigmoidDerAct(layers[i].a));
            }

            for (int i = 0; i < layers.Count - 1; i++)
            {
                layers[i].delta = layers[i].a.GetTranspose() * layers[i + 1].error;
            }

            for (int i = 1; i < layers.Count; i++)
            {
                layers[i - 1].w = layers[i - 1].w - ((layers[i - 1].delta / layers[0].a.GetLength(0)) * learningRate);
                layers[i].b = layers[i].b - (layers[i].error.AvgPerRow() * learningRate);
            }
        }


        public void BGD(double[,] inpAll, double[,] expAll, int bachSize, int eppochNr)
        {
            inputs = new Matrix(inpAll);
            expectedALL = new Matrix(expAll);

            for (int e = 0; e < eppochNr; e++)
            {
                double cost = 0;
                Shuffle();

                for (int i = 0; i < inputs.GetLength(0) / bachSize; i++)
                {
                    Matrix inp = new Matrix(bachSize, 784);
                    expectedOutputs = new Matrix(bachSize, 10);

                    for (int j = 0; j < bachSize; j++)
                    {
                        for (int q = 0; q < 784; q++)
                        {
                            inp[j, q] = inputs[i * bachSize + j, q];
                        }

                        for (int q = 0; q < 10; q++)
                        {
                            expectedOutputs[j, q] = expectedALL[i * bachSize + j, q];
                        }
                    }

                    layers[0].a = inp;

                    TrainAllZ();
                    //rtb.Text += "Eppoch nr: " + e + "\n" + "Bach Nr: " + i + "\n" + "Cost is" + GetCost() + "\n\n\n";
                    cost += GetCost();
                }

                rtb.Text += "Epoch nr " + e + "\n" + "Cost is: " + cost/bachSize + "\n\n\n";
            }
        }

        public void Shuffle()
        {
            for (int i = inputs.GetLength(0) - 1; i > 0; i--)
            {
                int r = rnd.Next(i);
                Swap(i, r);
            }
        }

        public void Swap(int a, int b)
        {
            for (int i = 0; i < inputs.GetLength(1); i++)
            {
                double aux = inputs[a, i];
                inputs[a, i] = inputs[b, i];
                inputs[b, i] = aux;
            }

            for (int i = 0; i < expectedALL.GetLength(1); i++)
            {
                double aux = expectedALL[a, i];
                expectedALL[a, i] = expectedALL[b, i];
                expectedALL[b, i] = aux;
            }
        }

        public Matrix SigmoidDerAct(Matrix a)
        {
            Matrix aux = a.ElemntWiseMult(Matrix.Ones(a.GetLength(0), a.GetLength(1)) - a);
            return aux;
        }

        public void RunWithoutTrain()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].z = layers[i - 1].a * layers[i - 1].w;

                layers[i].z = layers[i].z.AddByRow(layers[i].b);

                layers[i].a = Sigmoid(layers[i].z);
            }
        }

        public Matrix Predict(double[,] vals)
        {
            Matrix aux = new Matrix(layers[0].a);
            Matrix toP = new Matrix(vals);
            layers[0].a = toP;

            RunWithoutTrain();

            layers[0].a = aux;
            return new Matrix(layers[layers.Count - 1].a);
        }

        public Matrix Predict(Matrix vals)
        {
            Matrix toP = new Matrix(vals);
            layers[0].a = toP;

            RunWithoutTrain();
            return new Matrix(layers[layers.Count - 1].a);
        }

        public Matrix SigmoidDer(Matrix a)
        {
            Matrix aux = Sigmoid(a).ElemntWiseMult(Matrix.Ones(a.GetLength(0), a.GetLength(1)) - Sigmoid(a));
            return aux;
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private Matrix Sigmoid(Matrix a)
        {
            Matrix aux = new Matrix(a);

            for (int i = 0; i < aux.GetLength(0); i++)
            {
                for (int j = 0; j < aux.GetLength(1); j++)
                {
                    aux[i, j] = Sigmoid(aux[i, j]);
                }
            }

            return aux;
        }


        public double GetCost()
        {
            //return Cost.GetCost();
            return Cost.GetSum();
        }
    }
}
