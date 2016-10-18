# include <stdlib.h>
# include <stdio.h>


struct Neuron
{
	float bias;
	int nbInputs;
	float *weights;
};

struct Layer
{
	int nbNeurons;
	struct Neuron* neurons;
};

struct Network
{
	int nbLayers;
	int *nbNeurons;
	struct Layer* layers;
};

struct TrainingData
{
	float* trainingInputs;
	float desiredOutput;
};
