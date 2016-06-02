using System;
using System.Linq;
using NeuralNetworks.BackPropagation.Computation;
using NeuralNetworks.BackPropagation.Neurons;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace NeuralNetworks.BackPropagation.Networks
{
    public class NeuralNetworkFeatureMap
    {
        public IEnumerable<INeuron> Neurons { get; private set; }

        public NeuralNetworkFeatureMap(
            int numberOfNeurons, 
            IEnumerable<INeuron> inputs,
            DifferentiableFunction activationFunction,
            IRandomNumberGenerator weightGenerator)
        {
            this.Neurons = new ReadOnlyCollection<INeuron>(
                Enumerable.Range(0, numberOfNeurons)
                .Select(i => (INeuron)new Neuron(
                    activationFunction,
                    weightGenerator.Next(),
                    inputs.Select(n => new Synapse(n, weightGenerator.Next())).ToArray()))
                .ToList());
        }
    }
}