namespace NeuralNetworks.BackPropagation
{
    public class Synapse
    {
        public INeuron Neuron { get; private set; }
        public double Weight { get; set; }

        public Synapse(INeuron neuron, double initialWeight)
        {
            this.Neuron = neuron;
            this.Weight = initialWeight;
        }
    }
}