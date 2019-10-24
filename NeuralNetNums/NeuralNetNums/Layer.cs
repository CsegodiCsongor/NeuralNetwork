using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetNums
{
    public class Layer
    {
        public int neuronCount;

        public Matrix z;
        public Matrix a;
        public Matrix w;
        public Matrix b;

        public Matrix error;

        public Matrix delta;

        public double rDelta;
    }
}
