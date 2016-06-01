using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation
{
    public class Neuron : INeuron
    {
        public IEnumerable<Synapse> Synapses { get; protected set; }
        public DifferentiableFunction ActivationFunction { get; protected set; }
        public double Offset { get; private set; }
        public double Value { get; private set; }

        public Neuron(IEnumerable<Synapse> synapses, DifferentiableFunction activationFunction, double offset)
        {
            this.ActivationFunction = activationFunction;
            this.Offset = offset;
            this.Synapses = new ReadOnlyCollection<Synapse>(synapses.ToList());
            UpdateValue();
        }

        public void UpdateValue()
        {
            this.Value = ActivationFunction.Evaluate(
                this.Synapses.Sum(s => s.Weight * s.Neuron.Value) +
                this.Offset);
        }
    }
}
