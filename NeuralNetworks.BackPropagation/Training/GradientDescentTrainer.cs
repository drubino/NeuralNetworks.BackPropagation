using NeuralNetworks.BackPropagation.Computation;
using NeuralNetworks.BackPropagation.Networks;
using NeuralNetworks.BackPropagation.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation.Training
{
    public class GradientDescentTrainer
    {
        public NeuralNetwork Network { get; private set; }
        public double StepSize { get; private set; }

        public GradientDescentTrainer(NeuralNetwork network, double stepSize)
        {
            this.Network = network;
            this.StepSize = stepSize;
        }

        public IEnumerable<double> Train(IEnumerable<double> input, IEnumerable<double> targetOutput)
        {
            var actualOutput = this.Network.Execute(input);
            var targetOutputValues = targetOutput.ToList();
            var dError_dInput_Map = new Dictionary<INeuron, double>();

            var index = 0;
            foreach (var outputNeuron in this.Network.OutputNeurons)
            {
                var targetOutputValue = targetOutputValues[index++];
                var outputValue = outputNeuron.Value;
                var dError_dOutput = 2 * (outputValue - targetOutputValue);
                var dOutput_dInput = outputNeuron.ActivationFunction.EvaluateDerivative(outputValue);
                var dError_dInput = dError_dOutput * dOutput_dInput;
                dError_dInput_Map[outputNeuron] = dError_dInput;

                var dInput_dOffset = 1;
                var dError_dOffset = dError_dInput * dInput_dOffset;
                var dOffset = -this.StepSize * dError_dOffset;
                outputNeuron.Offset += dOffset;

                foreach (var synapse in outputNeuron.Synapses)
                {
                    var dInput_dWeight = outputValue;
                    var dError_dWeight = dError_dInput * dInput_dWeight;
                    var dWeight = -this.StepSize * dError_dWeight;
                    synapse.Weight += dWeight;
                }
            }

            var layersInReverse = this.Network.Layers.Reverse().ToList();
            foreach (var layer in layersInReverse)
            {
                foreach (var map in layer.FeatureMaps)
                {
                    foreach (var neuron in map.Neurons)
                    {
                        var dError_dInput = neuron.DownstreamSynapses.Sum(synapse => 
                            dError_dInput_Map[synapse.Parent] * synapse.Weight);
                        dError_dInput_Map[neuron] = dError_dInput;

                        var dInput_dOffset = 1;
                        var dError_dOffset = dError_dInput * dInput_dOffset;
                        var dOffset = -this.StepSize * dError_dOffset;
                        neuron.Offset += dOffset;

                        foreach (var synapse in neuron.Synapses)
                        {
                            var dInput_dWeight = neuron.Value;
                            var dError_dWeight = dError_dInput * dInput_dWeight;
                            var dWeight = -this.StepSize * dError_dWeight;
                            synapse.Weight += dWeight;
                        }
                    }
                }
            }

            return actualOutput;
        }
    }
}
