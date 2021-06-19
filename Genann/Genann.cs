using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace Genann {
	public class Ann {
		/// <summary>
		/// The action function of Genann.
		/// </summary>
		/// <param name="genann"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		public delegate double Action(double a);

		/// <summary>
		/// How many inputs, outputs, and hidden neurons.
		/// </summary>
		public struct Configure{
			public int Inputs;
			public int HiddenLayers;
			public int Hidden;
			public int Outputs;
		};

		protected const int LookupSize = 4096;
		protected const double _SigmoidDomMin = -15.0;
		protected const double _SigmoidDomMax = 15.0;
		protected static double _Interval;
		protected static double[] _Lookup = new double[LookupSize];

		protected Configure _Configure;

		protected Action _ActivationHidden;

		protected Action _ActivationOutput;

		protected int _TotalWeights;

		protected int _TotalNeurons;

		protected int _WeightIndex;

		protected int _OutputIndex;

		protected int _DeltaIndex;

		protected double[] _Data;

		/// <summary>
		/// Creates a new ann.
		/// </summary>
		/// <param name="Inputs"></param>
		/// <param name="HiddenLayers"></param>
		/// <param name="Hidden"></param>
		/// <param name="Outputs"></param>
		public Ann( int Inputs, int HiddenLayers, int Hidden, int Outputs ) {
			_Initialize( Inputs, HiddenLayers, Hidden, Outputs );
		}

		/// <summary>
		/// Sets weights randomly.
		/// </summary>
		public void Randomize( int seed = 0 ) {
			Random random = new Random(seed);
			for ( int i = 0; i < _TotalWeights; i++ ) {
				double r = random.NextDouble();
				_Data[_WeightIndex + i] = r - 0.5;
			}
		}

		/// <summary>
		/// Runs the feedforward algorithm to calculate the ann's output.
		/// </summary>
		/// <returns></returns>
		public IEnumerable<double> Run(params double[] Inputs ) {
			int ret;
			int w = _WeightIndex;
			int o = _OutputIndex + _Configure.Inputs;
			int i = _OutputIndex;
			int h, j, k;

			// Copy the inputs to the scratch area, where we also store each neuron's
			// output, for consistency.This way the first layer isn't a special case.
			Array.Copy(Inputs, 0, _Data, _OutputIndex, _Configure.Inputs);

			if ( _Configure.HiddenLayers == 0 ) {
				ret = o;
				for ( j = 0; j < _Configure.Outputs; j++ ) {
					double sum = _Data[w++] * -1.0;
					for ( k = 0; k < _Configure.Inputs; k++ ) {
						sum += _Data[w++] * _Data[i + k];
					}
					_Data[o++] = _ActivationOutput(sum);
				}

				return _Data.Skip(ret).Take(_Configure.Outputs);
			}

			// Figure input layer
			for ( j = 0; j < _Configure.Hidden; j++ ) {
				double sum = _Data[w++] * -1.0;
				for (k = 0; k < _Configure.Inputs; k++ ) {
					sum += _Data[w++] * _Data[i + k];
				}
				_Data[o++] = _ActivationHidden(sum);
			}

			i += _Configure.Inputs;

			// Figure hidden layers, if any.
			for ( h = 1; h < _Configure.HiddenLayers; h++ ) {
				for ( j = 0; j < _Configure.Hidden; j++ ) {
					double sum = _Data[w++] * -1.0;
					for ( k = 0; k < _Configure.Hidden; k++ ) {
						sum += _Data[w++] * _Data[i + k];
					}
					_Data[o++] = _ActivationHidden(sum);
				}
				i += _Configure.Hidden;
			}

			ret = o;

			// Figure output layer.
			for ( j = 0; j < _Configure.Outputs; j++ ) {
				double sum = _Data[w++] * -1.0;
				for ( k = 0; k < _Configure.Hidden; k++ ) {
					sum += _Data[w++] * _Data[i + k];
				}
				_Data[o++] = _ActivationOutput(sum);
			}

			return _Data.Skip(ret).Take(_Configure.Outputs);
		}

		/// <summary>
		/// Does a single backprop update.
		/// </summary>
		/// <param name="Inputs"></param>
		/// <param name="DesiredOutputs"></param>
		/// <param name="LearningRate"></param>
		public void Train(double[] Inputs, double[] DesiredOutputs, double LearningRate) {
			// To begin with, we must run the network forward.
			TrainPrepareInputs( Inputs );

			// First set the output layer deltas.
			TrainPrepareDesiredOutputs( DesiredOutputs );

			// Set hidden layer deltas, start on last layer and work backwards.
			// Note that loop is skipped in the case of hidden_layers == 0.
			TrainProcess(LearningRate);
		}

		/// <summary>
		/// Prepare inputs for a training.
		/// </summary>
		/// <param name="Inputs"></param>
		public void TrainPrepareInputs( params double[] Inputs ) {
			// To begin with, we must run the network forward.
			Run(Inputs);
		}

		/// <summary>
		/// Prepare desire outputs for a training.
		/// </summary>
		/// <param name="DesiredOutputs"></param>
		public void TrainPrepareDesiredOutputs( params double[] DesiredOutputs) {
			int j;
			// First set the output layer deltas.
			{
				int o = _OutputIndex + _Configure.Inputs + _Configure.Hidden * _Configure.HiddenLayers;
				int d = _DeltaIndex + _Configure.Hidden * _Configure.HiddenLayers;
				int t = 0;

				// Set output layer deltas.
				if (_ActivationOutput == _ActLinear)
				{
					for (j = 0; j < _Configure.Outputs; j++)
					{
						_Data[d++] = DesiredOutputs[t++] - _Data[o++];

					}
				}
				else
				{
					for (j = 0; j < _Configure.Outputs; j++)
					{
						_Data[d++] = (DesiredOutputs[t] - _Data[o]) * _Data[o] * (1.0 - _Data[o]);
						o++;
						t++;
					}
				}
			}
		}

		/// <summary>
		/// Process a training.
		/// </summary>
		/// <param name="LearningRate"></param>
		public void TrainProcess( double LearningRate ) {
			int h, j, k;
			// Set hidden layer deltas, start on last layer and work backwards.
			// Note that loop is skipped in the case of hidden_layers == 0.
			for (h = _Configure.HiddenLayers - 1; h >= 0; h--)
			{
				// Find first output and delta in this layer.
				int o = _OutputIndex + _Configure.Inputs + (h * _Configure.Hidden);
				int d = _DeltaIndex + (h * _Configure.Hidden);
				// Find first delta in following layer (which may be hidden or output).
				int dd = _DeltaIndex + ((h + 1) * _Configure.Hidden);
				// Find first weight in following layer (which may be hidden or output).
				int ww = _WeightIndex + ((_Configure.Inputs + 1) * _Configure.Hidden) + ((_Configure.Hidden + 1) * _Configure.Hidden * (h));

				for (j = 0; j < _Configure.Hidden; j++)
				{
					double delta = 0;

					for (k = 0; k < (h == _Configure.HiddenLayers - 1 ? _Configure.Outputs : _Configure.Hidden); k++)
					{
						double forwardDelta = _Data[dd + k];
						int windex = k * (_Configure.Hidden + 1) + (j + 1);
						double forwardWeight = _Data[ww + windex];
						delta += forwardDelta * forwardWeight;
					}

					_Data[d] = _Data[o] * (1.0 - _Data[o]) * delta;
					d++;
					o++;
				}
			}

			// Train the outputs.
			{
				// Find first output delta.
				int d = _DeltaIndex + _Configure.Hidden * _Configure.HiddenLayers;

				// Find first weight to first output delta.
				int w = _WeightIndex + (_Configure.HiddenLayers != 0
					? ((_Configure.Inputs + 1) * _Configure.Hidden + (_Configure.Hidden + 1) * _Configure.Hidden * (_Configure.HiddenLayers - 1))
					: (0));

				// Find first output in previous layer.
				int i = _OutputIndex + (_Configure.HiddenLayers != 0
					? (_Configure.Inputs + (_Configure.Hidden) * (_Configure.HiddenLayers - 1))
					: 0);

				// Set output layer weights.
				for (j = 0; j < _Configure.Outputs; j++)
				{
					_Data[w++] += _Data[d] * LearningRate * -1.0;
					for (k = 1; k < (_Configure.HiddenLayers != 0 ? _Configure.Hidden : _Configure.Inputs) + 1; k++)
					{
						_Data[w++] += _Data[d] * LearningRate * _Data[i + (k - 1)];
					}
					d++;
				}
			}

			// Train the hidden layers.
			{
				for (h = _Configure.HiddenLayers - 1; h >= 0; h--)
				{
					// Find first delta in this layer.
					int d = _DeltaIndex + (h * _Configure.Hidden);

					// Find first input to this layer.
					int i = _OutputIndex + (h != 0
						? (_Configure.Inputs + _Configure.Hidden * (h - 1))
						: 0);

					// Find first weight to this layer.
					int w = _WeightIndex + (h != 0
						? ((_Configure.Inputs - 1) * _Configure.Hidden + (_Configure.Hidden + 1) * (_Configure.Hidden) * (h - 1))
						: 0);

					for (j = 0; j < _Configure.Hidden; j++)
					{
						_Data[w++] += _Data[d] * LearningRate * -1.0;
						for (k = 1; k < (h == 0 ? _Configure.Inputs : _Configure.Hidden) + 1; k++)
						{
							_Data[w++] += _Data[d] * LearningRate * _Data[i + (k - 1)];
						}
						d++;
					}
				}
			}
		}

		/// <summary>
		/// Saves the ann.
		/// </summary>
		public void Dump( Stream ostream ) {
			StreamWriter writer = new StreamWriter(ostream);
			writer.WriteLine("{0} {1} {2} {3}", _Configure.Inputs, _Configure.HiddenLayers, _Configure.Hidden, _Configure.Outputs);
			for ( int i = 0; i < _TotalWeights; i++ ) {
				writer.WriteLine("{0}", _Data[_WeightIndex + i]);
			}
			writer.Flush();
		}

		/// <summary>
		/// Load ANN from stream saved with Dump method.
		/// </summary>
		public static Ann Load( Stream istream ) {
			StreamReader reader = new StreamReader(istream);
			string[] header = reader.ReadLine().Split(' ');
			Ann genann = new Ann( int.Parse(header[0]), int.Parse(header[1]), int.Parse(header[2]), int.Parse(header[3]) );
			for ( int i = 0; i < genann._TotalWeights; i++ ) {
				genann._Data[genann._WeightIndex + i] = double.Parse(reader.ReadLine());
			}
			return genann;
		}

		protected void _Initialize(int Inputs, int HiddenLayers, int Hidden, int Outputs)
		{
			if (HiddenLayers > 0 && Hidden < 1)
			{
				Hidden = 1;
			}

			int hiddenWeights = (HiddenLayers != 0) ? (Inputs + 1) * Hidden + (HiddenLayers - 1) * (Hidden + 1) * Hidden : 0;
			int outputWeights = ((HiddenLayers != 0) ? (Hidden + 1) : (Inputs + 1)) * Outputs;
			int totalWeights = (hiddenWeights + outputWeights);

			int totalNeurons = (Inputs + Hidden * HiddenLayers + Outputs);

			_Configure.Inputs = Inputs;
			_Configure.HiddenLayers = HiddenLayers;
			_Configure.Hidden = Hidden;
			_Configure.Outputs = Outputs;

			_TotalWeights = totalWeights;
			_TotalNeurons = totalNeurons;

			_Data = new double[totalWeights + totalNeurons + (totalNeurons - Inputs)];
			_WeightIndex = 0;
			_OutputIndex = _WeightIndex + totalWeights;
			_DeltaIndex = _OutputIndex + totalNeurons;

			Randomize();

			_ActivationHidden = _ActSigmoidCached;
			_ActivationOutput = _ActSigmoidCached;

			_InitSigmoidLookup();
		}

		protected void _InitSigmoidLookup() {
			const double f = (_SigmoidDomMax - _SigmoidDomMin) / LookupSize;
			_Interval = LookupSize / (_SigmoidDomMax - _SigmoidDomMin);
			for ( int i = 0; i < LookupSize; i++ ) {
				_Lookup[i] = _ActSigmoid(_SigmoidDomMin + f * i);
			}
		}

		protected double _ActSigmoid(double a) {
			if (a < -45.0) return 0;
			if (a > 45.0) return 1;
			return 1.0 / (1 + Math.Exp(-a));
		}

		protected double _ActSigmoidCached(double a)
		{
			if (a < _SigmoidDomMin) return _Lookup[0];
			if (a >= _SigmoidDomMax) return _Lookup[LookupSize - 1];
			int j = (int)((a - _SigmoidDomMin) * _Interval + 0.5);
			if (j >= LookupSize) return _Lookup[LookupSize - 1];
			return _Lookup[j];
		}

		protected double _ActThreshold(double a) => a > 0 ? 1 : 0;

		protected double _ActLinear(double a) => a;
	}
}