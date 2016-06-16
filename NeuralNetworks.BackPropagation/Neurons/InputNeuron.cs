using NeuralNetworks.BackPropagation.Computation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation.Neurons
{
    public class InputNeuron : INeuron
    {
        private double value;
        private HashSet<Synapse> downstreamSynapses = new HashSet<Synapse>();
        public IEnumerable<Synapse> Synapses { get; private set; }
        public IEnumerable<Synapse> DownstreamSynapses { get; private set; }
        public DifferentiableFunction ActivationFunction { get; private set; }
        public double Offset { get; set; }
        public double Value
        {
            get { return this.value; }
            set
            {
                if (this.value != value)
                {
                    this.value = value;
                    Reset();
                }
            }
        }

        public InputNeuron(double initialValue = 0)
        {
            this.value = initialValue;
            this.ActivationFunction = new DifferentiableFunction(x => x);
            this.Synapses = Enumerable.Empty<Synapse>();
            this.DownstreamSynapses = Enumerable.Empty<Synapse>();
        }

        public void AddDownstreamSynapse(Synapse synapse)
        {
            this.downstreamSynapses.Add(synapse);
        }

        public void Reset()
        {
            foreach (var synapse in this.downstreamSynapses)
                synapse.Reset();
        }
    }
}
