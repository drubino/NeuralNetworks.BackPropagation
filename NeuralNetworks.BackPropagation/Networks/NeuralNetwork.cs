using NeuralNetworks.BackPropagation.Computation;
using NeuralNetworks.BackPropagation.Neurons;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation.Networks
{
    public class NeuralNetwork
    {
        private DifferentiableFunction activationFunction;
        private IRandomNumberGenerator weightGenerator;
        private List<NeuralNetworkLayer> layers = new List<NeuralNetworkLayer>();
        private ReadOnlyCollection<InputNeuron> inputNeurons = new ReadOnlyCollection<InputNeuron>(new List<InputNeuron>());
        private ReadOnlyCollection<INeuron> outputNeurons = new ReadOnlyCollection<INeuron>(new List<INeuron>());
        private bool inputsAreAssigned = false;
        private bool outputsAreAssigned = false;

        public ICollection<InputNeuron> InputNeurons { get { return this.inputNeurons; } }
        public ICollection<INeuron> OutputNeurons { get { return this.outputNeurons; } }
        public IEnumerable<NeuralNetworkLayer> Layers { get { return this.layers; } }

        public NeuralNetwork(
            DifferentiableFunction activationFunction, 
            IRandomNumberGenerator weightGenerator)
        {
            this.activationFunction = activationFunction;
            this.weightGenerator = weightGenerator;
        }

        public IEnumerable<double> Execute(params double[] inputs)
        {
            return Execute((IEnumerable<double>)inputs);
        }

        public IEnumerable<double> Execute(IEnumerable<double> inputs)
        {
            var currentInputs = inputs.ToList();
            if (!this.inputsAreAssigned || !this.outputsAreAssigned)
                throw new InvalidOperationException("The inputs and outputs of the network have not been initialized yet.");
            if (currentInputs.Count() != this.InputNeurons.Count())
                throw new ArgumentException($"The number of inputs do not match the number of input neurons");

            var index = 0;
            foreach (var currentInput in currentInputs)
            {
                var inputNeuron = this.inputNeurons[index++];
                inputNeuron.Value = currentInput;
            }

            var result = this.outputNeurons.Select(n => n.Value).ToList();
            return result;
        }

        public NeuralNetwork AddInputs(int numberOfInputs)
        {
            if (this.inputsAreAssigned)
                throw new InvalidOperationException("The inputs are already assigned.");

            this.inputNeurons = new ReadOnlyCollection<InputNeuron>(
                Enumerable.Range(0, numberOfInputs)
                .Select(i => new InputNeuron())
                .ToList());

            this.inputsAreAssigned = true;
            return this;
        }

        public NeuralNetworkLayer AddLayer()
        {
            if (!this.inputsAreAssigned)
                throw new InvalidOperationException("A layer cannot be added because the inputs are not assigned.");
            if (this.outputsAreAssigned)
                throw new InvalidOperationException("A layer cannot be added because the outputs are already assigned.");

            var layerInputs = GetNeuronsFromPreviousLayer();
            var layer = new NeuralNetworkLayer(this, layerInputs, this.activationFunction, this.weightGenerator);
            this.layers.Add(layer);

            return layer;
        }

        public void AddOutputs(int numberOfNeurons)
        {
            if (!this.inputsAreAssigned)
                throw new InvalidOperationException("The outputs cannot be added because the inputs are not assigned.");

            var inputs = GetNeuronsFromPreviousLayer();
            this.outputNeurons = new ReadOnlyCollection<INeuron>(
                Enumerable.Range(0, numberOfNeurons)
                .Select(i => (INeuron)new Neuron(
                    new DifferentiableFunction(x => x),
                    this.weightGenerator.Next(),
                    inputs.Select(n => new Synapse(n, this.weightGenerator.Next())).ToArray()))
                .ToList());

            this.outputsAreAssigned = true;
        }

        private IEnumerable<INeuron> GetNeuronsFromPreviousLayer()
        {
            var previousNeurons = this.layers.Any() ?
                this.layers.Last().FeatureMaps.SelectMany(m => m.Neurons) :
                this.InputNeurons;

            return previousNeurons;
        }
    }
}
