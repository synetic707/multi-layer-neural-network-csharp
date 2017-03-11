using System;
using MathNet.Numerics.LinearAlgebra;

namespace multilayerNN
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var layer1 = new NeuronLayer(4, 3);
            var layer2 = new NeuronLayer(1, 4);

            var neuralNetwork = new NeuralNetwork(layer1, layer2);

            double[,] trainingSetInputs =
            {
                {0, 0, 1},
                {0, 1, 1},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 0},
                {1, 1, 1},
                {0, 0, 0}
            };

            double[,] trainingSetOutputs =
            {
                {0},
                {1},
                {1},
                {1},
                {1},
                {0},
                {0}
            };


            Console.WriteLine("Start training ...");

            // Train the neural network using a training set.
            // Do it 10,000 times
            neuralNetwork.Train(trainingSetInputs, trainingSetOutputs, 10000);
            Console.WriteLine("End training ...\n\n");

            // Predict
            Console.WriteLine("Considering new situation [1, 1, 0] -> ?\n");

            Matrix<double> hiddenLayer;
            Matrix<double> result;

            neuralNetwork.Think(new double[,] {{1, 1, 0}}, out hiddenLayer, out result);

            Console.WriteLine(result);
            Console.ReadKey();
        }
    }
}