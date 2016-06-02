using NeuralNetworks.BackPropagation.Computation;
using NeuralNetworks.BackPropagation.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation.Networks
{
    public class NeuralNetworkLayer
    {
        private IEnumerable<INeuron> inputs;
        private DifferentiableFunction activationFunction;
        private IRandomNumberGenerator weightGenerator;
        private List<NeuralNetworkFeatureMap> featureMaps = new List<NeuralNetworkFeatureMap>();

        public NeuralNetwork NeuralNetwork { get; private set; }
        public IEnumerable<NeuralNetworkFeatureMap> FeatureMaps { get { return this.featureMaps; } }

        public NeuralNetworkLayer(
            NeuralNetwork neuralNetwork,
            IEnumerable<INeuron> inputs, 
            DifferentiableFunction activationFunction, 
            IRandomNumberGenerator weightGenerator)
        {
            this.NeuralNetwork = neuralNetwork;
            this.inputs = inputs;
            this.activationFunction = activationFunction;
            this.weightGenerator = weightGenerator;
        }

        public NeuralNetworkLayer AddFeatureMap(int numberOfNeurons)
        {
            this.featureMaps.Add(
                new NeuralNetworkFeatureMap(
                    numberOfNeurons, 
                    this.inputs, 
                    this.activationFunction,
                    this.weightGenerator));

            return this;
        }
    }
}
