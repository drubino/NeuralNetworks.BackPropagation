using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation
{
    public class InputNeuron : INeuron
    {
        public double Value { get; private set; }

        public InputNeuron(double initialValue)
        {
            this.Value = initialValue;
        }

        public void UpdateValue() { }
    }
}
