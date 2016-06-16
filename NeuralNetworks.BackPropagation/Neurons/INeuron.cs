using NeuralNetworks.BackPropagation.Computation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation.Neurons
{
    public interface INeuron
    {
        double Value { get; }
        double Offset { get; set; }
        IEnumerable<Synapse> Synapses { get; }
        IEnumerable<Synapse> DownstreamSynapses { get; }
        DifferentiableFunction ActivationFunction { get; }
        void AddDownstreamSynapse(Synapse synapse);
        void Reset();
    }
}
