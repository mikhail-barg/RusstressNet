# RusstressNet
A .Net port of https://github.com/MashaPo/russtress

[![Build status](https://ci.appveyor.com/api/projects/status/tiqminujri769s0l/branch/master?svg=true)](https://ci.appveyor.com/project/mikhail-barg/russtressnet/branch/master) [![NuGet](https://img.shields.io/nuget/v/RusstressNet.svg)](https://www.nuget.org/packages/RusstressNet/)

Description from oiriginal repo:
> The tool based on LSTM predicts stress position in each word in russian text depending on the word context. For more details about the tool see [«Automated Word Stress Detection in Russian»](http://www.aclweb.org/anthology/W/W17/W17-4104.pdf), EMNLP-2017, Copenhagen, Denmark.

Neural network model is converted from original Keras/TF model with [keras2onnx](https://github.com/onnx/keras-onnx), see [conversion script](https://github.com/mikhail-barg/RusstressNet/blob/master/RusstressNet/convert.py). Inference run by [ONNX Runtime](https://github.com/microsoft/onnxruntime). 

Simplified and streamlined pre- and post-processing code.

## Usage
```c#
using RusstressNet;

using (AccentModel model = new AccentModel())
{
	string accentedText = model.SetAccentForText(text));
}
```

Also see [console model executor](https://github.com/mikhail-barg/RusstressNet/blob/master/RusstressExecutor/Program.cs).

## Thread-safety

After creation, instances of `AccentModel` are thread-safe (because ONNX InferrenseSession [is thread-safe](https://github.com/Microsoft/onnxruntime/issues/114)), so you generally need a single instance of it.
