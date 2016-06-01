using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation
{
    public interface INeuron
    {
        double Value { get; }
        void UpdateValue();
    }
}
