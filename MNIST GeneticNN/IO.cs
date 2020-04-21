using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace MNIST_GeneticNN
{
    class IO
    {
        static readonly string BasePath = @"C:\Users\gwflu\source\repos\MNIST GeneticNN\MNIST GeneticNN\WBs";
        public static bool Running = false;
        public static NN Read(int num)
        {
            NN nn = new NN();
            if (Running) { throw new Exception("Already accessing file"); }
            Running = true;
            FileStream fs = new FileStream(BasePath + "\\" + num.ToString() + ".txt", FileMode.Open, FileAccess.Read, FileShare.None);
            StreamReader sr = new StreamReader(fs);
            string text = sr.ReadToEnd();
            sr.Close(); fs.Close();
            string[] split = text.Split(' ');

            int numlayers = int.Parse(split[0]);
            nn.Layers = new List<Layer>();

            int iterator = 1;
            for (int j = 0; j < numlayers; j++)
            {
                int length = int.Parse(split[iterator]); iterator++;
                int inputlength = int.Parse(split[iterator]); iterator++;
                nn.Layers.Add(new Layer(length, inputlength));
                for (int i = 0; i < length; i++)
                {
                    for (int ii = 0; ii < inputlength; ii++)
                    {
                        nn.Layers[j].Weights[i, ii] = double.Parse(split[iterator]);
                        iterator++;
                    }
                    nn.Layers[j].Biases[i] = double.Parse(split[iterator]);
                    iterator++;
                }
            }
            Running = false;
            return nn;
        }
        public static void Write(NN nn, int num)
        {
            FileStream fs = new FileStream(BasePath + "\\" + num.ToString() + ".txt", FileMode.Create, FileAccess.Write, FileShare.None);
            StreamWriter sw = new StreamWriter(fs);
            sw.Write(nn.NumLayers + " ");
            foreach (Layer l in nn.Layers)
            {
                sw.Write(l.Length + " " + l.InputLength + " ");
                for (int i = 0; i < l.Length; i++)
                {
                    for (int ii = 0; ii < l.InputLength; ii++)
                    {
                        sw.Write(l.Weights[i, ii] + " ");
                    }
                    sw.Write(l.Biases[i] + " ");
                }
            }
            sw.Close(); fs.Close();
        }
    }
}
