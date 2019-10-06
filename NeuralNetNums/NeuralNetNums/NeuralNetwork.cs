using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetNums
{
    public class NeuralNetwork
    {
        public double learningRate = 0.03;

        public List<Layer> layers;

        public Matrix inputs;
        public Matrix expectedOutputs;

        public NeuralNetwork(double[,] inputs, double[,] expectedOutputs, int[] layerCounts)
        {
            layers = new List<Layer>();
            this.inputs = new Matrix(inputs);
            this.expectedOutputs = new Matrix(expectedOutputs);

            layers.Add(new Layer());
            layers[0].neuronCount = layerCounts[0];

            layers[0].w = Matrix.Randoms(layerCounts[0], layerCounts[1]);
            //layers[0].w = Matrix.Ones(layerCounts[0], layerCounts[1]);

            layers[0].b = new Matrix(1, layerCounts[0]);
            layers[0].a = new Matrix(inputs);

            for (int i = 1; i < layerCounts.Length - 1; i++)
            {
                layers.Add(new Layer());
                layers[i].neuronCount = layerCounts[i];

                layers[i].w = Matrix.Randoms(layerCounts[i], layerCounts[i + 1]);
                //layers[i].w = Matrix.Ones(layerCounts[i], layerCounts[i + 1]);

                layers[i].b = new Matrix(1, layerCounts[i]);
            }

            layers.Add(new Layer());
            layers[layers.Count - 1].neuronCount = layerCounts[layerCounts.Length - 1];
            layers[layers.Count - 1].b = new Matrix(1, layerCounts[layers.Count - 1]);
        }


        public void Run()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].z =layers[i - 1].w.GetTranspose() * layers[i - 1].a;
                layers[i].a = Sigmoid(layers[i].z);
            }

            Train();
        }

        public void Train()
        {
            layers[layers.Count - 1].delta = (layers[layers.Count - 1].a - expectedOutputs).ElemntWiseMult(layers[layers.Count - 1].a - expectedOutputs);
            layers[layers.Count - 1].rDelta = layers[layers.Count - 1].delta.GetSum();


            layers[layers.Count - 2].delta = layers[layers.Count - 2].w * SigmoidDer(layers[layers.Count - 1].z) * (2 * (layers[layers.Count - 1].a - expectedOutputs)).GetTranspose();
            layers[layers.Count - 2].rDelta = layers[layers.Count - 2].delta.GetSum();

            for(int i = layers.Count - 3; i > 0; i--)
            {
                //layers[i].delta = layers[i].w * SigmoidDer(layers[i + 1].z) * layers[i + 1].delta;
                layers[i].delta = layers[i].w * SigmoidDer(layers[i + 1].z) * layers[i + 1].rDelta;
                layers[i].rDelta = layers[i].delta.GetSum();
            }


            //layers[layers.Count - 2].w -= learningRate * ( layers[layers.Count-2].a * SigmoidDer(layers[layers.Count - 1].z).GetTranspose() * (2 * (layers[layers.Count - 1].a - expectedOutputs)).GetTranspose());

            for(int i=0; i < layers.Count-2; i++)
            {
                //layers[i].w -=learningRate * (layers[i].a * SigmoidDer(layers[i + 1].z) * layers[i + 1].delta);
                layers[i].w -= learningRate * (layers[i].a * SigmoidDer(layers[i + 1].z) * layers[i + 1].rDelta);
            }
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
            return layers[layers.Count - 1].rDelta;
        }
    }
}
