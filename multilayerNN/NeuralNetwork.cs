using MathNet.Numerics.LinearAlgebra;

namespace multilayerNN
{
    public class NeuralNetwork
    {
        private readonly NeuronLayer _layer1;
        private readonly NeuronLayer _layer2;

        public NeuralNetwork(NeuronLayer layer1, NeuronLayer layer2)
        {
            _layer1 = layer1;
            _layer2 = layer2;
        }

        private Matrix<double> Sigmoid(Matrix<double> x, bool derivative)
        {
            if (derivative)
                return x.PointwiseMultiply(1 - x);
            return 1 / (1 + 1 / x.PointwiseExp());
        }

        public void Train(double[,] trainingInput, double[,] trainingOutput, int trainingIterations)
        {
            var mtrainingInput = BuildMatrixFromArray(trainingInput);
            var mtrainingOutput = BuildMatrixFromArray(trainingOutput);

            for (var i = 0; i < trainingIterations; i++)
            {
                Matrix<double> outputFromLayer1;
                Matrix<double> outputFromLayer2;

                Think(trainingInput, out outputFromLayer1, out outputFromLayer2);

                var layer2Error = mtrainingOutput - outputFromLayer2;
                var layer2Delta = Dot(layer2Error, Sigmoid(outputFromLayer2, false));

                var layer1Error = Dot(layer2Delta, _layer2.SynapticWeights);
                var layer1Delta = Dot(layer1Error, Sigmoid(outputFromLayer1, true));

                var layer1Adjustment = mtrainingInput.TransposeThisAndMultiply(layer1Delta);
                var layer2Adjustment = Dot(outputFromLayer1.Transpose(), layer2Delta);

                _layer1.SynapticWeights += layer1Adjustment.Transpose();
                _layer2.SynapticWeights += layer2Adjustment.Transpose();
            }
        }

        public void Think(double[,] inputs, out Matrix<double> outputFromLayer1, out Matrix<double> outputFromLayer2)
        {
            var minputs = BuildMatrixFromArray(inputs);

            outputFromLayer1 = Sigmoid(minputs * _layer1.SynapticWeights.Transpose(), false);
            outputFromLayer2 = Sigmoid(outputFromLayer1 * _layer2.SynapticWeights.Transpose(), false);
        }

        private Matrix<double> Dot(Matrix<double> matrixOne, Matrix<double> matrixTwo)
        {
            if (matrixOne.ColumnCount == matrixTwo.ColumnCount && matrixOne.RowCount == matrixTwo.RowCount)
                return matrixOne.PointwiseMultiply(matrixTwo);

            return matrixOne.Multiply(matrixTwo);
        }

        private Matrix<double> BuildMatrixFromArray(double[,] array)
        {
            return Matrix<double>.Build.DenseOfArray(array);
        }
    }
}