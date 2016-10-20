# include <stdlib.h>
# include <stdio.h>


struct Neuron
{
	float bias;
	int nbInputs;
	float *weights;
	float nabla_b;
	float delta_nabla_b;
	float *nabla_w;
	float *delta_nabla_w;
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
