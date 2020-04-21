using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_GeneticNN
{
    static class Quicksort
    {
        private static void Swap<T>(T[] array, int a, int b)
        {
            var intermediary = array[a];
            array[a] = array[b];
            array[b] = intermediary;
        }
        //Pushes the better networks further down the array
        private static int Partition(NN[] array, int min, int max)
        {
            NN pivot = array[(min + (max - min) / 2)];
            int i = min - 1;
            int j = max + 1;
            do
            {
                //While pivot wins to array[i]
                do { i++; } while (array[i].PercCorrect < pivot.PercCorrect);
                //And loses to array[j]
                do { j--; } while (array[j].PercCorrect > pivot.PercCorrect);
                if (i >= j) { return j; }
                Swap(array, i, j);
            } while (true);
        }
        public static void Quick(NN[] array, int min, int max)
        {
            if (min < max)
            {
                int pivot = Partition(array, min, max);
                Quick(array, min, pivot);
                Quick(array, pivot + 1, max);
            }
        }
    }
}
