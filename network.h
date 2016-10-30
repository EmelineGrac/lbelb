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
	float activation;
	float z;
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
	int* desiredOutput;
};

float sigmoid(float z);
float sigmoid_prime(float z);
float cost_derivative(float output_activation, int y);
double random_normal(void);
int highest(float *result, int size);
float* feedforward(struct Network n, int iLayer, float *inputsVect);
int test(struct Network n, float *inputsVect);
int evaluate(struct Network n, float **inputs, int *outputs, size_t len);
void backprop(struct Network *n, float *trainingInputs,	int* desiredOutput);
void update_mini_batch(struct Network *n,
		       struct TrainingData *k,
		       struct TrainingData* k_end,
		       float eta);
void SGD(struct Network *n,
	 struct TrainingData *td,
	 size_t size_td,
	 int epochs,
	 int mini_batch_size,
	 float eta);
void initNeuron(struct Neuron *_neuron, float _bias, int _nbInputs);
void initLayer(struct Layer *_layer, int _nbNeurons, int _nbInputs);
void initNetwork(struct Network *_network, int _nbLayers, int *_nbNeurons);
void array_print(int *begin, int *end);
void printNetwork(struct Network n);
void open(struct Network *n, char fileName[]);
void write(struct Network n, char fileName[]);
void freeMemoryNetwork(struct Network* n);
void randomInit(struct Network *n);
int* indexOutputToVector(int index, size_t len);
