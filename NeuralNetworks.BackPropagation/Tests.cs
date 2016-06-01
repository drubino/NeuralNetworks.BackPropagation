using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NeuralNetworks.BackPropagation
{
    [TestClass]
    public class Tests
    {
        [TestMethod]
        public void TestDifferentiableFunction()
        {
            var epsilon = .0001;
            var function = new DifferentiableFunction(x => 1 / (1 + Math.Exp(-x)));
            Func<double, double> derivative = x => function.Evaluate(x) * (1 - function.Evaluate(x));

            foreach (var x in Enumerable.Range(-50, 101).Select(x => x / 10))
            {
                var expectedDerivative = function.EvaluateDerivative(x);
                var actualDerivative = derivative(x);
                var difference = Math.Abs(expectedDerivative - actualDerivative);
                Assert.IsTrue(difference < epsilon);
            }
        }
    }
}
