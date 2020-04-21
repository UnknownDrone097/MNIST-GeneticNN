using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_GeneticNN
{
    class Genetics
    {
        public bool Run = false;
        public bool Reset = false;
        public double PercCorrect = 0;
        int trialnum = 0;
        public int PopSize { get; set; }
        public double SurvivalProportion { get; set; }
        public double SurvivalRateChange { get; set; }
        public double SurvivalFloor { get; set; }
        public double MutationProportion { get; set; }
        public double NumberOfMutations { get; set; }
        NN[] NNs { get; set; }

        /// <summary>
        /// Initializes the genetics class
        /// </summary>
        /// <param name="load">Whether or not to load weights and biases from file</param>
        /// <param name="ps">Population size</param>
        /// <param name="sp">Likelyhood of survival between generations (per individual)</param>
        /// <param name="src">How much more likely the strong are to survive than the weak (linearly)</param>
        /// <param name ="sf">The lowest chance of survival possible</param>
        /// <param name="mp">How likely mutations are to occur (in an individual)</param>
        /// <param name="nm">Number of mutations per subject</param>
        public Genetics(bool load, int ps, double sp, double src, double sf, double mp, int nm)
        {
            PopSize = ps; SurvivalProportion = sp; SurvivalRateChange = src; MutationProportion = mp;
            SurvivalFloor = sf; NumberOfMutations = nm;
            //Initialize population
            if (!load) { GeneratePopulation(PopSize); }
            else { NNs = Load(); }
        }
        public void Save()
        {
            //Save population
            for (int i = 0; i < PopSize; i++)
            {
                IO.Write(NNs[i], i);
            }
        }
        public NN[] Load()
        {
            var nns = new NN[PopSize];
            for (int i = 0; i < PopSize; i++) 
            { 
                nns[i] = IO.Read(i);
            }
            return nns;
        }
        public void Evaluate(int batchSize, bool testing)
        {
            var Solutions = new int[batchSize];
            var Questions = new double[batchSize][,];
            for (int i = 0; i < batchSize; i++)
            {
                Solutions[i] = Reader.ReadNextLabel();
                Questions[i] = Reader.ReadNextImage();
            }
            var tasks = new Task[PopSize * batchSize];
            for (int i = 0; i < PopSize; i++)
            {
                for (int ii = 0; ii < batchSize; ii++)
                {
                    int i2 = i; int ii2 = ii;
                    tasks[(i2 * batchSize) + ii2] = Task.Run(() => NNs[i2].Run(Questions[ii2], Solutions[ii2], testing));
                }
            }
            Task.WaitAll(tasks);
        }
      
        //Quicksort!
        public void Tournament()
        {
            //Sort NNs
            Quicksort.Quick(NNs, 0, NNs.Length - 1);

            double perccorrect = 0;
            foreach (NN nn in NNs) 
            { 
                perccorrect += nn.PercCorrect;
            }
            perccorrect /= PopSize; 

            if (Reset) { trialnum = 0; Reset = false; }
            trialnum++;
            PercCorrect = (PercCorrect * ((trialnum) / (trialnum + 1d))) + (perccorrect * (1d / trialnum));            
        }
        public void Propegate()
        {
            //Select
            //NNs are stored fitest to least fit in this list
            //(opposite of the array)
            var OldNNs = new List<NN>();
            
            double LikelyhoodOfSurvival = SurvivalProportion;
            var r = new Random();
            for (int i = PopSize - 1; i >= 0; i--)
            {
                //Surive if randomly falls within proportion
                if (r.NextDouble() < LikelyhoodOfSurvival) { OldNNs.Add(NNs[i]); }
                //Chance of survival decreases with lower fitness down to some floor
                if (LikelyhoodOfSurvival > SurvivalFloor) { LikelyhoodOfSurvival -= SurvivalRateChange; }
            }
            //Crossover
            NNs = new NN[PopSize];
            for (int i = 0; i < OldNNs.Count; i++) { NNs[i] = OldNNs[i]; }
            for (int i = PopSize - 1; i >= 0; i--)
            {
                if (!(NNs[i] is null)) { continue; }
                //Generate child
                NNs[i] = Mutation(Crossover(OldNNs[r.Next(0, OldNNs.Count)], OldNNs[r.Next(0, OldNNs.Count)], true), MutationProportion);
            }
        }
        public NN Crossover(NN parent1, NN parent2, bool side)
        {
            var r = new Random();
            NN child = new NN();
            child.Init();
            for (int i = 0; i < parent1.Layers.Count; i++)
            {
                int crossoverPointX = r.Next(0, parent1.Layers[i].Weights.GetLength(0));
                int crossoverPointY = r.Next(0, parent1.Layers[i].Weights.GetLength(1));
                for (int ii = 0; ii < parent1.Layers[i].Weights.GetLength(0); ii++)
                {
                    for (int iii = 0; iii < parent1.Layers[i].Weights.GetLength(1); iii++)
                    {
                        double gene = parent1.Layers[i].Weights[ii, iii];
                        if (ii > crossoverPointX && iii > crossoverPointY)
                        {
                            gene = parent2.Layers[i].Weights[ii, iii];
                        }
                        child.Layers[i].Weights[ii, iii] = gene;
                    }
                    double bgene = parent1.Layers[i].Biases[ii];
                    if (ii > crossoverPointX)
                    {
                        bgene = parent2.Layers[i].Biases[ii];
                    }
                    child.Layers[i].Biases[ii] = bgene;
                }
            }
            return child;
        }
        public NN Mutation(NN patient, double probability)
        {
            var r = new Random();
            if (r.NextDouble() > probability) { return patient; }
            for (int i = 0; i < NumberOfMutations; i++)
            {
                int mutationLayer = r.Next(0, patient.Layers.Count);
                int mutationPointX = r.Next(0, patient.Layers[mutationLayer].Biases.GetLength(0));
                //50% to mutate weight or bias
                if (r.NextDouble() < .5)
                {

                    //"Flip the bit" so to speak to mutate
                    patient.Layers[mutationLayer].Biases[mutationPointX]
                       = 1d - patient.Layers[mutationLayer].Biases[mutationPointX];
                }
                else
                {
                    int mutationPointY = r.Next(0, patient.Layers[mutationLayer].Weights.GetLength(1));
                    //"Flip the bit" so to speak to mutate
                    patient.Layers[mutationLayer].Weights[mutationPointX, mutationPointY]
                        = 1d - patient.Layers[mutationLayer].Weights[mutationPointX, mutationPointY];
                }
            }
            return patient;
        }
        void GeneratePopulation(int popsize)
        {
            NNs = new NN[popsize];
            for (int i = 0; i < popsize; i++)
            {
                var nn = new NN();
                nn.Init();
                NNs[i] = nn;
            }
            Save();
        }
    }
}
