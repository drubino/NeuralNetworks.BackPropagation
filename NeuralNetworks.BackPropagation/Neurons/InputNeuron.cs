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
