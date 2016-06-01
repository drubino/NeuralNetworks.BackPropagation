namespace NeuralNetworks.BackPropagation
{
    public class Synapse
    {
        private INeuron parent;
        private double weight;

        public INeuron Neuron { get; private set; }
        public double Weight
        {
            get { return this.weight; }
            set
            {
                if (this.weight != value)
                {
                    this.weight = value;
                    Reset();
                }
            }
        }

        public Synapse(INeuron neuron, double initialWeight)
        {
            this.Neuron = neuron;
            this.Neuron.AddDownstreamSynapse(this);
            this.weight = initialWeight;
        }

        public void AddParent(INeuron parent)
        {
            this.parent = parent;
        }

        public void Reset()
        {
            if (parent != null)
                this.parent.Reset();
        }
    }
}