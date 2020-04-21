using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace MNIST_GeneticNN
{
    class NN
    {
        static int Resolution = 28;
        //Must be 1 for now
        public int NumLayers = 2;
        public int INCount = 28;
        public int NCount = 10;
        public int ONCount = 10;
        public List<Layer> Layers { get; set; }
        public static double Momentum = .9;
        public static double LearningRate = .000146;
        public static double RMSDecay = .9;
        public static bool UseMomentum = false;
        public static bool UseRMSProp = true;

        public double TrialNum = 0;
        public double PercCorrect = 0;
        public double Error = 0;
        public int Guess { get; set; }

        /// <summary>
        /// Create a NN of the specified statuses
        /// </summary>
        /// <param name="learningrate">Speed of learning</param>
        /// <param name="momentum">Whether to use [Nesterov] weight momentum</param>
        /// <param name="userms">Whether to use Root Mean Square propegation</param>
        /// <param name="numlayers">How many layers are in the model</param>
        /// <param name="incount">Number of neurons in the input layer</param>
        /// <param name="hidcount">Number of neurons in the hidden layer</param>
        /// <param name="outcount">Number of neurons in the output layer</param>
        public NN(double learningrate, int numlayers, int incount, int hidcount, int outcount)
        {
            NumLayers = numlayers; 
            INCount = incount; NCount = hidcount; ONCount = outcount; LearningRate = learningrate;
        }
        public NN() { }
        public void Init()
        {
            Layers = new List<Layer>();
            int count = NCount;
            int lowercount = Resolution;
            Random r = new Random();
            for (int i = 0; i < NumLayers; i++)
            {
                if (i != 0) { lowercount = Layers[i - 1].Length; count = NCount; }
                if (i == 0) { count = INCount; }
                if (i == NumLayers - 1) { count = ONCount; }
                Layers.Add(new Layer(count, lowercount));
                for (int j = 0; j < count; j++)
                {
                    for (int jj = 0; jj < lowercount; jj++)
                    {
                        //Weights initialized to a random number in range (-1/(2 * lowercount^2) - 1/(2 * lowercount^2))
                        //This is Lecun initialization
                        Layers[i].Weights[j, jj] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (double)(lowercount * lowercount));
                    }
                }
            }
        }
        public void Run(double[,] image, int correct, bool testing)
        {
            double[,] input = new double[28, 28];
            //Deepclone?
            for (int i = 0; i < 28; i++) { for (int ii = 0; ii < 28; ii++) { input[i, ii] = image[i, ii]; } }

            //Forward
            for (int i = 0; i < Layers.Count; i++)
            {
                if (i == 0) { Layers[i].Calculate(input); continue; }
                Layers[i].Calculate(Layers[i - 1].Values, i == Layers.Count - 1);
            }
            //Report values
            Guess = -1; double certainty = -5; double error = 0;
            for (int i = 0; i < ONCount; i++)
            {
                if (Layers[Layers.Count - 1].Values[i] > certainty) { Guess = i; certainty = Layers[Layers.Count - 1].Values[i]; }
                error += ((i == correct ? 1d : 0d) - Layers[Layers.Count - 1].Values[i]) * ((i == correct ? 1d : 0d) - Layers[Layers.Count - 1].Values[i]);
            }
            TrialNum++;

            PercCorrect = (PercCorrect * ((TrialNum) / (TrialNum + 1))) + ((Guess == correct) ? (1 / (TrialNum)) : 0d);
            Error = (Error * ((TrialNum) / (TrialNum + 1))) + (error * (1 / (TrialNum)));
        }
    }
    class ActivationFunctions
    {
        public static double Tanh(double number)
        {
            return (Math.Pow(Math.E, 2 * number) - 1) / (Math.Pow(Math.E, 2 * number) + 1);
        }
        public static double TanhDerriv(double number)
        {
            return (1 - number) * (1 + number);
        }
        public static double[] Normalize(double[] array)
        {
            double mean = 0;
            double stddev = 0;
            //Calc mean of data
            foreach (double d in array) { mean += d; }
            mean /= array.Length;
            //Calc std dev of data
            foreach (double d in array) { stddev += (d - mean) * (d - mean); }
            stddev /= array.Length;
            stddev = Math.Sqrt(stddev);
            //Prevent divide by zero b/c of sigma = 0
            if (stddev == 0) { stddev = .000001; }
            //Calc zscore
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (array[i] - mean) / stddev;
            }

            return array;
        }
        public static double[,] Normalize(double[,] array, int depth, int count)
        {
            double[] smallarray = new double[depth * count];
            int iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count; ii++)
                {
                    smallarray[iterator] = array[i, ii];
                    iterator++;
                }
            }
            smallarray = Normalize(smallarray);
            iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count; ii++)
                {
                    array[i, ii] = smallarray[iterator];
                    iterator++;
                }
            }
            return array;
        }
    }
    class Layer
    {
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        public double[] Values { get; set; }
        public double[] Errors { get; set; }
        public int Length { get; set; }
        public int InputLength { get; set; }
        public Layer(int length, int inputlength)
        {
            Length = length;
            InputLength = inputlength;

            Weights = new double[Length, InputLength];
            Biases = new double[Length];
        }
        public void Backprop(Layer output)
        {
            Errors = new double[Length];
            for (int k = 0; k < output.Length; k++)
            {
                for (int j = 0; j < Length; j++)
                {
                    Errors[j] += output.Weights[k, j] * ActivationFunctions.TanhDerriv(output.Values[k]) * output.Errors[k];
                }
            }
        }
        public void Backprop(int correct)
        {
            Errors = new double[Length];
            for (int i = 0; i < Length; i++)
            {
                Errors[i] = 2d * ((i == correct ? 1d : 0d) - Values[i]);
            }
        }
        public void Calculate(double[] input, bool output)
        {
            Values = new double[Length];
            for (int k = 0; k < Length; k++)
            {
                for (int j = 0; j < InputLength; j++)
                {
                    Values[k] += Weights[k, j] * input[j];
                }
                if (!output)
                {
                    Values[k] += Biases[k];
                    Values[k] = ActivationFunctions.Tanh(Values[k]);
                }
                else { Values[k] = Values[k]; }
            }
        }
        public void Calculate(double[,] input)
        {
            double[] input2 = new double[input.Length];
            int iterator = 0;
            foreach (double d in input) { input2[iterator] = d; iterator++; }
            Calculate(input2, false);
        }
    }
}