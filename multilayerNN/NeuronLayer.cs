using System;
using MathNet.Numerics.LinearAlgebra;

namespace multilayerNN
{
    public class NeuronLayer
    {
        public Matrix<double> SynapticWeights;

        public NeuronLayer(int numberOfNeurons, int numberOfInputsPerNeuron)
        {
            SynapticWeights = CreateMatrixWithRandomNumbers(numberOfNeurons, numberOfInputsPerNeuron);
        }

        private Matrix<double> CreateMatrixWithRandomNumbers(int rowCount, int columnCount)
        {
            var rand = new Random();
            var matrix = new double[rowCount, columnCount];

            for (var i = 0; i < rowCount; i++)
            for (var j = 0; j < columnCount; j++)
            {
                var maxValue = 1.0;
                var minValue = -1.0;

                var betweenMinusOneToOne = rand.NextDouble() * (maxValue - minValue) + minValue;
                matrix[i, j] = betweenMinusOneToOne;
            }

            return Matrix<double>.Build.DenseOfArray(matrix);
        }
    }
}