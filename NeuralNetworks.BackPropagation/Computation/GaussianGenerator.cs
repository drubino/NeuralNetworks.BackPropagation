using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation.Computation
{
    public class GaussianGenerator : IRandomNumberGenerator
    {
        [ThreadStatic]
        private static Random random = new Random();
        private double mean;
        private double standardDeviation;

        public GaussianGenerator(double mean = 0, double standardDeviation = 1)
        {
            this.mean = mean;
            this.standardDeviation = standardDeviation;
        }

        public double Next()
        {
            //these are uniform(0,1) random doubles
            var uniform1 = random.NextDouble(); 
            var uniform2 = random.NextDouble();

            //random normal(0,1)
            var randomNormal = 
                Math.Sqrt(-2.0 * Math.Log(uniform1)) *
                Math.Sin(2.0 * Math.PI * uniform2);

            //random normal(mean,std^2)
            return this.mean + (this.standardDeviation * randomNormal); 
        }
    }
}
