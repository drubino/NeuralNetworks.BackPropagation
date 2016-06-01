using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BackPropagation
{
    public class DifferentiableFunction
    {
        private Func<double[], double> function;
        private static double epsilon = 0.0000001;

        public DifferentiableFunction(Func<double, double> function)
        {
            this.function = x => function(x[0]);
        }

        public DifferentiableFunction(Func<double[], double> function)
        {
            this.function = function;
        }

        public double Evaluate(double input)
        {
            return this.function(new[] { input });
        }

        public double Evaluate(double[] input)
        {
            return this.function(input);
        }

        public double EvaluateDerivative(double input)
        {
            return EvaluateDerivative(new[] { input }, 0);
        }

        public double EvaluateDerivative(double[] input, int dimension)
        {
            var count = 0;
            var step = epsilon;
            double difference = 0;
            while (count < 10)
            {
                var left = input.ToArray();
                left[dimension] = left[dimension] - step;
                var leftResult = Evaluate(left);

                var right = input.ToArray();
                right[dimension] = right[dimension] + step;
                var rightResult = Evaluate(right);

                difference = rightResult - leftResult;
                if (Math.Abs(difference) < epsilon)
                    return difference / (2 * step);

                step = step / 2;
                count = count + 1;
            }

            return difference / (2 * step);
        }
    }
}
