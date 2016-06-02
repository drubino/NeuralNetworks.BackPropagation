using NeuralNetworks.BackPropagation.Computation;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation.Neurons
{
    public class Neuron : INeuron
    {
        private bool hasValue;
        private double value;
        private double offset;
        private HashSet<Synapse> downstreamSynapses = new HashSet<Synapse>();

        public IEnumerable<Synapse> Synapses { get; protected set; }
        public DifferentiableFunction ActivationFunction { get; protected set; }
        public double Offset
        {
            get { return this.offset; }
            set
            {
                if (this.offset != value)
                {
                    this.offset = value;
                    Reset();
                }
            }
        }

        public double Value
        {
            get
            {
                if (!this.hasValue)
                {
                    this.hasValue = true;
                    this.value = ActivationFunction.Evaluate(
                        this.Synapses.Sum(s => s.Weight * s.Neuron.Value) +
                        this.Offset);
                }

                return this.value;
            }
        }

        public Neuron(DifferentiableFunction activationFunction, double offset, params Synapse[] synapses)
        {
            this.ActivationFunction = activationFunction;
            this.offset = offset;
            this.Synapses = new ReadOnlyCollection<Synapse>(synapses.ToList());
            foreach (var synapse in this.Synapses)
                synapse.AddParent(this);
        }

        public void Reset()
        {
            if (!this.hasValue)
                return;

            this.hasValue = false;
            foreach (var synapse in this.downstreamSynapses)
                synapse.Reset();
        }

        public void AddDownstreamSynapse(Synapse synapse)
        {
            this.downstreamSynapses.Add(synapse);
        }
    }
}
