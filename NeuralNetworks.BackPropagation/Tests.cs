using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using NeuralNetworks.BackPropagation.Computation;
using NeuralNetworks.BackPropagation.Neurons;
using NeuralNetworks.BackPropagation.Networks;
using NeuralNetworks.BackPropagation.Training;

namespace NeuralNetworks.BackPropagation
{
    [TestClass]
    public class Tests
    {
        #region TestDifferentiableFunction

        [TestMethod]
        public void TestDifferentiableFunction()
        {
            var epsilon = .0001;
            var function = new DifferentiableFunction(x => 1 / (1 + Math.Exp(-x)));
            Func<double, double> derivative = x => function.Evaluate(x) * (1 - function.Evaluate(x));

            foreach (var x in Enumerable.Range(-50, 101).Select(x => x / 10))
            {
                var expectedDerivative = function.EvaluateDerivative(x);
                var actualDerivative = derivative(x);
                var difference = Math.Abs(expectedDerivative - actualDerivative);
                Assert.IsTrue(difference < epsilon);
            }
        }

        #endregion //TestDifferentiableFunction

        #region TestNeuronReset

        [TestMethod]
        public void TestNeuronReset()
        {
            var activationFunction = new DifferentiableFunction(x => x);

            var input1 = new InputNeuron(0);
            var input2 = new InputNeuron(0);
            var synapse111 = new Synapse(input1, 1);
            var synapse112 = new Synapse(input2, 1);
            var synapse121 = new Synapse(input1, 1);
            var synapse122 = new Synapse(input2, 1);
            var neuron1 = new Neuron(activationFunction, 0, synapse111, synapse112);
            var neuron2 = new Neuron(activationFunction, 0, synapse121, synapse122);
            var synapse211 = new Synapse(neuron1, 1);
            var synapse212 = new Synapse(neuron2, 1);
            var synapse221 = new Synapse(neuron1, 1);
            var synapse222 = new Synapse(neuron2, 1);
            var output1 = new Neuron(activationFunction, 0, synapse211, synapse212);
            var output2 = new Neuron(activationFunction, 0, synapse221, synapse222);

            input1.Value = 1;
            input2.Value = 1;
            var o1 = output1.Value;
            var o2 = output2.Value;

            synapse111.Weight = 2;
            neuron1.Offset = 1;
            o1 = output1.Value;
            o2 = output2.Value;
        }

        #endregion //TestNeuronReset

        #region TestNeuralNetwork

        [TestMethod]
        public void TestNeuralNetwork()
        {
            var activationFunction = new DifferentiableFunction(x => 1 / (1 + Math.Exp(-x)));
            var weightGenerator = new GaussianGenerator();
            var network = new NeuralNetwork(activationFunction, weightGenerator);

            network
                .AddInputs(2)
                .AddLayer()
                    .AddFeatureMap(2)
                    .AddFeatureMap(2)
                    .NeuralNetwork
                .AddOutputs(2);

            var result = network.Execute(1, 1);
            var result2 = network.Execute(2, 2);
            var result3 = network.Execute(1, 1);

        }

        #endregion //TestNeuralNetwork

        #region TestGradientDescentTrainer

        [TestMethod]
        public void TestGradientDescentTrainer()
        {
            var activationFunction = new DifferentiableFunction(x => 1 / (1 + Math.Exp(-x)));
            var weightGenerator = new GaussianGenerator();
            var network = new NeuralNetwork(activationFunction, weightGenerator);
            var trainer = new GradientDescentTrainer(network, .01);

            network
                .AddInputs(1)
                .AddLayer()
                    .AddFeatureMap(10)
                    .NeuralNetwork
                .AddLayer()
                    .AddFeatureMap(10)
                    .NeuralNetwork
                .AddOutputs(1);

            Func<double, double, double> train = (input, targetOutput) => 
                trainer.Train(new double[] { input }, new double[] { targetOutput }).Single();
            Func<double, double> execute = input =>
                network.Execute(input).First();

            var random = new Random();
            foreach (var value in Enumerable.Range(0, 100000))
            {
                var input = random.NextDouble();
                var targetOutput = Math.Round(input);
                train(input, targetOutput);
            }
        }

        #endregion //TestGradientDescentTrainer
    }
}
