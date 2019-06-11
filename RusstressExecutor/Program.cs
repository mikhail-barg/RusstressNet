using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RusstressNet;

namespace RusstressExecutor
{
    internal sealed class Program
    {
        static void Main(string[] args)
        {
            using (AccentModel model = new AccentModel())
            {
                Console.WriteLine("Enter text: ");
                while (true)
                {
                    Console.Write(">>> ");
                    string text = Console.ReadLine();
                    if (text == String.Empty)
                    {
                        break;
                    }
                    Console.WriteLine(model.SetAccentForText(text));
                }
            }
        }
    }
}
