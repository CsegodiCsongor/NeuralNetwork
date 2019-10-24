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
            //layers[0].w = Matrix.Ones(layerCounts[0], layerCounts[1]);

            layers[0].b = Matrix.Ones(1, layerCounts[0]);

            layers[0].a = new Matrix(inputs);
            //layers[0].z = new Matrix(inputs);

            for (int i = 1; i < layerCounts.Length - 1; i++)
            {
                layers.Add(new Layer());
                layers[i].neuronCount = layerCounts[i];

                layers[i].w = Matrix.Randoms(layerCounts[i], layerCounts[i + 1]);
                //layers[i].w = Matrix.Ones(layerCounts[0], layerCounts[1]);

                layers[i].b = Matrix.Ones(1, layerCounts[i]);
            }

            layers.Add(new Layer());
            layers[layers.Count - 1].neuronCount = layerCounts[layerCounts.Length - 1];

            layers[layers.Count - 1].b = Matrix.Ones(1, layerCounts[layers.Count - 1]);
        }


        public void Run()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].z = layers[i - 1].a * layers[i - 1].w;

                layers[i].z = layers[i].z.AddByRow(layers[i].b);

                layers[i].a = Sigmoid(layers[i].z);
            }

            Train1();
        }

        public void Train()
        {
            /* Cost = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(layers[layers.Count - 1].a - expectedOutputs);

             layers[layers.Count - 1].delta = 2 * (layers[layers.Count - 1].a - expectedOutputs);
             layers[layers.Count - 1].rDelta = layers[layers.Count - 1].delta.GetSum();


             //layers[layers.Count - 2].delta = layers[layers.Count - 2].w * (SigmoidDer(layers[layers.Count - 1].z).ElemntWiseMult(2 * (layers[layers.Count - 1].a - expectedOutputs)));
             //layers[layers.Count - 2].rDelta = layers[layers.Count - 2].delta.GetSum();

             for(int i = layers.Count - 2; i > 0; i--)
             {
                 //layers[i].delta = layers[i].w * SigmoidDer(layers[i + 1].z) * layers[i + 1].delta;
                 //layers[i].delta = SigmoidDer(layers[i + 1].z) * layers[i].w.GetTranspose() * layers[i + 1].rDelta;

                 layers[i].delta = SigmoidDer(layers[i + 1].z).ByColMult(layers[i + 1].delta.AvgPerRow()) * layers[i].w.GetTranspose();
                 //layers[i].delta = layers[i + 1].delta.ElemntWiseMult(SigmoidDer(layers[i + 1].z));

                 layers[i].rDelta = layers[i].delta.GetSum();
             }


             //layers[layers.Count - 2].w -= learningRate * ( layers[layers.Count-2].a * SigmoidDer(layers[layers.Count - 1].z).GetTranspose() * (2 * (layers[layers.Count - 1].a - expectedOutputs)).GetTranspose());

             for(int i=0; i < layers.Count-2; i++)
             {
                 //layers[i].w -=learningRate * (layers[i].a * SigmoidDer(layers[i + 1].z) * layers[i + 1].delta);
                 //layers[i].w -= learningRate * (layers[i].a.GetTranspose() * SigmoidDer(layers[i + 1].z) * layers[i + 1].rDelta);
                 layers[i].w -= learningRate * (layers[i].a.GetTranspose() * (SigmoidDer(layers[i + 1].z).ElemntWiseMult(layers[i + 1].delta)));
             }

             for(int i=1; i < layers.Count - 1; i++)
             {
                 //layers[i].b -= learningRate * (SigmoidDer(layers[i].z).AvgPerRow().GetTranspose() * layers[i].rDelta);

                 layers[i].b -= learningRate * (SigmoidDer(layers[i].z).ElemntWiseMult(layers[i].delta)).AvgPerRow().GetTranspose();
             }

             layers[layers.Count-1].b -= learningRate * (SigmoidDer(layers[layers.Count - 1].z).ElemntWiseMult(2 * (layers[layers.Count - 1].a - expectedOutputs))).AvgPerRow().GetTranspose();*/

            Cost = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(layers[layers.Count - 1].a - expectedOutputs);

            //layers[layers.Count - 1].error = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(SigmoidDer(layers[layers.Count-1].z));
            layers[layers.Count - 1].error = (layers[layers.Count - 1].a - expectedOutputs);

            for (int i = layers.Count - 2; i >= 0; i--)
            {
                //layers[i].error = (layers[i].w * layers[i + 1].error.ElemntWiseMult(SigmoidDer(layers[i + 1].z)).GetTranspose()).GetTranspose();
                //Matrix g = layers[i + 1].error.AvgPerCol();
                layers[i].error = (layers[i + 1].error * layers[i].w.GetTranspose()).ElemntWiseMult(SigmoidDer(layers[i].z));

                if (layers[i].delta != null)
                {
                    layers[i].delta += (layers[i].a.GetTranspose() * layers[i + 1].error) / 100;
                    //layers[i].delta = layers[i].a * layers[i].error.GetTranspose();
                }
                else
                {
                    layers[i].delta = (layers[i].a.GetTranspose() * layers[i + 1].error) / 100;
                    //layers[i].delta = layers[i].a * layers[i].error.GetTranspose();
                }
            }

            for (int i = 0; i < layers.Count - 2; i++)
            {
                layers[i].w -= layers[i].delta * learningRate;
                layers[i].b -= layers[i].delta.GetTranspose().AvgPerCol() * learningRate;
            }
        }

        public void Train1()
        {
            Cost = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(layers[layers.Count - 1].a - expectedOutputs);

            layers[layers.Count - 1].delta = -1 * ((layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(SigmoidDer(layers[layers.Count - 1].z)));
            layers[layers.Count - 1].rDelta = layers[layers.Count - 1].delta.GetSum();


            //layers[layers.Count - 2].delta = layers[layers.Count - 2].w * (SigmoidDer(layers[layers.Count - 1].z).ElemntWiseMult(2 * (layers[layers.Count - 1].a - expectedOutputs)));
            //layers[layers.Count - 2].rDelta = layers[layers.Count - 2].delta.GetSum();

            for (int i = layers.Count - 2; i > 0; i--)
            {
                //layers[i].delta = layers[i].w * SigmoidDer(layers[i + 1].z) * layers[i + 1].delta;
                //layers[i].delta = SigmoidDer(layers[i + 1].z) * layers[i].w.GetTranspose() * layers[i + 1].rDelta;

                //layers[i].delta = SigmoidDer(layers[i + 1].z).ByColMult(layers[i + 1].delta.AvgPerRow()) * layers[i].w.GetTranspose();
                //layers[i].delta = layers[i + 1].delta.ElemntWiseMult(SigmoidDer(layers[i + 1].z));

                layers[i].delta = (layers[i + 1].delta * layers[i].w).ElemntWiseMult(SigmoidDer(layers[i].z));

                layers[i].rDelta = layers[i].delta.GetSum();
            }


            //layers[layers.Count - 2].w -= learningRate * ( layers[layers.Count-2].a * SigmoidDer(layers[layers.Count - 1].z).GetTranspose() * (2 * (layers[layers.Count - 1].a - expectedOutputs)).GetTranspose());

            for (int i = 0; i < layers.Count - 2; i++)
            {
                //layers[i].w -=learningRate * (layers[i].a * SigmoidDer(layers[i + 1].z) * layers[i + 1].delta);
                //layers[i].w -= learningRate * (layers[i].a.GetTranspose() * SigmoidDer(layers[i + 1].z) * layers[i + 1].rDelta);
                //layers[i].w -= learningRate * (layers[i].a.GetTranspose() * (SigmoidDer(layers[i + 1].z).ElemntWiseMult(layers[i + 1].delta)));
                layers[i].w -= learningRate * (layers[i].a.GetTranspose() * layers[i + 1].delta);
            }

            for (int i = 1; i < layers.Count - 1; i++)
            {
                //layers[i].b -= learningRate * (SigmoidDer(layers[i].z).AvgPerRow().GetTranspose() * layers[i].rDelta);

                layers[i].b -= learningRate * (SigmoidDer(layers[i].z).ElemntWiseMult(layers[i].delta)).AvgPerRow().GetTranspose();
            }

            layers[layers.Count - 1].b -= learningRate * (SigmoidDer(layers[layers.Count - 1].z).ElemntWiseMult(2 * (layers[layers.Count - 1].a - expectedOutputs))).AvgPerRow().GetTranspose();
        }

        public void Train2()
        {
            errors = new Matrix[layers.Count - 1];
            deltas = new Matrix[layers.Count - 1];

            Cost = null;

            //List<Matrix> errors = new List<Matrix>();
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                Matrix expected = new Matrix(1, expectedOutputs.GetLength(1));
                layers[0].a = new Matrix(1, inputs.GetLength(1));

                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    layers[0].a[0, j] = inputs[i, j];
                }

                for (int j = 0; j < expectedOutputs.GetLength(1); j++)
                {
                    expected[0, j] = expectedOutputs[i, j];
                }

                RunWithoutTrain();

                //Cost = (layers[layers.Count - 1].a - expected).ElemntWiseMult(layers[layers.Count - 1].a - expected);
                if (Cost == null)
                {
                    Cost = (layers[layers.Count - 1].a - expected).ElemntWiseMult(layers[layers.Count - 1].a - expected);
                }
                else
                {
                    Cost += (layers[layers.Count - 1].a - expected).ElemntWiseMult(layers[layers.Count - 1].a - expected);
                }

                layers[layers.Count - 1].error = 2 * (layers[layers.Count - 1].a - expected);

                if (errors[errors.Length - 1] == null)
                {
                    errors[errors.Length - 1] = new Matrix(layers[layers.Count - 1].error);
                }
                else
                {
                    errors[errors.Length - 1] += layers[layers.Count - 1].error;
                }

                for (int j = layers.Count - 2; j > 0; j--)
                {
                    //layers[j].error = (layers[j].w * layers[j + 1].error.GetTranspose()).ElemntWiseMult(SigmoidDerAct(layers[j + 1].a));
                    layers[j].error = ((layers[j].w.ByColMult(SigmoidDerAct(layers[j + 1].a))) * layers[j + 1].error.GetTranspose()).GetTranspose();

                    if (errors[j - 1] == null)
                    {
                        errors[j - 1] = new Matrix(layers[j].error);
                    }
                    else
                    {
                        errors[j - 1] += layers[j].error;
                    }
                }

                for (int j = 0; j < layers.Count - 1; j++)
                {
                    layers[j].delta = (layers[j].a.GetTranspose() * layers[j + 1].error).ByColMult(SigmoidDerAct(layers[j + 1].a));

                    if (deltas[j] == null)
                    {
                        deltas[j] = new Matrix(layers[j].delta);
                    }
                    else
                    {
                        deltas[j] += layers[j].delta;
                    }
                }
            }


            for (int j = 0; j < deltas.Length; j++)
            {
                layers[j].w = layers[j].w - ((deltas[j] / 10000) * learningRate);
                layers[j + 1].b = layers[j + 1].b - ((errors[j] / 10000) * learningRate);
            }

            //Cost = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(layers[layers.Count - 1].a - expectedOutputs);
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
