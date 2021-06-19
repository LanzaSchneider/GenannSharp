using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace Genann
{
	class Program
	{
		public static void Main( string[] args ) {
			const int scale = 100;
			Random random = new Random((int)DateTime.Now.Ticks);
			Ann genann = new Ann(2, 1, 2, 1);
			genann.Randomize(random.Next());

			// Training
			for ( int i = 0; i < 8000; i++ ) {
				for ( int j = 0, len = scale / 2; j < len; j++ ) {
					int a = j;
					int b = j;
					int c = a + b;
					genann.TrainPrepareInputs( 1.0 * a / scale, 1.0 * b / scale );
					genann.TrainPrepareDesiredOutputs( 1.0 * c / scale );
					genann.TrainProcess( 4 );
				}
			}

			// Test Dump / Load
			{
				// Dump genann in a memory block.
				MemoryStream mstream = new MemoryStream();
				genann.Dump(mstream);
				Console.WriteLine($"Dumpdata size: { mstream.Length }");

				// Load genann from a memory block.
				mstream.Seek(0, SeekOrigin.Begin);
				genann = Ann.Load(mstream);

				// Close memory stream.
				mstream.Close();
			}

			// Run result
			const int amount = 1000;
			List<int> correctRates = new List<int>();
			correctRates.Add(0);
			for ( int i = 0; i < amount; i++ ) {
				int a = random.Next() % (scale / 2);
				int b = random.Next() % (scale / 2);
				int c = (int)Math.Round(scale * genann.Run(1.0 * a / scale, 1.0 * b / scale).First());
				int c2 = a + b;
				{
					int diff = Math.Abs(c - c2);

					while (diff >= correctRates.Count) correctRates.Add(correctRates[correctRates.Count - 1]);

					for ( int j = diff; j < correctRates.Count; j++ ) {
						correctRates[j] += 1;
					}
				}
				Console.Write("{0} + {1} = {2} [{3}]\n",
					a, b, c, c2
					);
			}

			// Correct rate
			for ( int i = 0, len = correctRates.Count; i < len; i++ ) {
				Console.Write("Correct rate with diff({0}): {1}%\n", i, 100.0 * correctRates[i] / amount);
			}

			Console.ReadLine();
		}
	}
}
