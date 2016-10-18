# include <stdlib.h>
# include <stdio.h>
# include <math.h>

# include "network.h"

float sigmoid(float z){
	return 1.0 / (1.0 + exp(-z));
}

int highest(float* result, int size){
// return the index of the output which has the highest score
	int res = 0;
	float max = result[0];
	for(int i = 0; i < size; i++){
		if (result[i] > max){
			max = result[i];
			res = i;
		}
	}
	return res;
}

float* feedforward(struct Network n, int iLayer, float* inputsVect){
// return the output of the network if inputsVect is input
// input layer = 0
	if (iLayer == n.nbLayers)
		return inputsVect;
	else{
		int j = 0;
		float res = 0;
		float* outputsVect = calloc(n.layers[iLayer].nbNeurons,sizeof(float));
		struct Neuron nr = n.layers[iLayer].neurons[0];
		while (j < n.layers[iLayer].nbNeurons){
			res = 0;
			nr = n.layers[iLayer].neurons[j];
			for(int k = 0; k < nr.nbInputs; k++){
				res += inputsVect[k] * nr.weights[k];}

			outputsVect[j] = sigmoid(res + nr.bias);
			j++;
		}
		return feedforward(n, iLayer + 1, outputsVect);
	}
}

int test(struct Network n, float* inputsVect){
	return highest(feedforward(n, 1, inputsVect),n.layers[n.nbLayers-1].nbNeurons);
}

int evaluate(struct Network n, float **inputs,
		 int *outputs, size_t len){
/* Return the number of test inputs for which the neural
 * network outputs the correct result. Note that the neural
 * network's output is assumed to be the index of whichever
 * neuron in the final layer has the highest activation.*/
	int res = 0;
	for(size_t i = 0; i < len; i++){
		if (highest(feedforward(n, 1, inputs[i]),
		n.layers[n.nbLayers-1].nbNeurons) == outputs[i])
			res++;
	}
	return res;
}

/*void SGD(struct Network* n, struct TrainingData* td, int epochs,
int mini_batch_size, float eta){


}*/


void initNeuron(struct Neuron* _neuron, float _bias, int _nbInputs)
{
	_neuron->bias = _bias;
	_neuron->nbInputs = _nbInputs;
	float *_weights = malloc(_nbInputs * sizeof(float));
	float rn;
	float rn_max = 3.0;
	for(int i = 0; i < _nbInputs; i++){
		// rn will be the same
		rn = (float)rand()/(float)(RAND_MAX/rn_max);
		_weights[i] = rn;
	}
	_neuron->weights = _weights;
}
void initLayer(struct Layer* _layer, int _nbNeurons, int _nbInputs)
{
	_layer->nbNeurons = _nbNeurons;
	struct Neuron *_neurons = (struct Neuron*)
(malloc(_nbNeurons * sizeof(struct Neuron)));
	/*int *begin = _neurons;
	int *end = _neurons + _nbNeurons;
	for (; begin < end;begin++)
		{
		 initNeuron(*begin, 0, 0);
		}*/
	float rn = 0.0;
	float rn_max = 1.0;
	for(int i = 0; i < _nbNeurons; i++)
		{
		rn = (float)rand()/(float)(RAND_MAX/rn_max);
		initNeuron(&_neurons[i], rn, _nbInputs);
		}
	_layer->neurons = _neurons;


}
void initNetwork(struct Network* _network, int _nbLayers, int *_nbNeurons)
{
	_network->nbLayers = _nbLayers;
	_network->nbNeurons = _nbNeurons;
	struct Layer *_layers = (struct Layer*)
(malloc(_nbLayers * sizeof(struct Layer)));
/*	int *begin = _layers;
	int *end = _layers + _nbLayers;
	int *begin1 = _nbNeurons;
	//int *end1 = _nbNeurons + _nbLayers;
	for (; begin < end; begin++, begin1++)
		{
		initLayer(*begin, *begin1);
		}*/
		int _nbInputs = 0;// first layer weights?
		for(int i = 0; i < _nbLayers; i++)
		{
		initLayer(&_layers[i], _nbNeurons[i],_nbInputs);
		_nbInputs = _layers[i].nbNeurons;
		}
	_network->layers = _layers;
}


void array_print(int *begin, int *end)
{
  int line = 0;
  for (; begin != end; ++begin) {
    if (line > 72) {
      printf("|`|\n");
      line = 0;
    }
    line += printf("| %4d ", *begin);
  }
  printf("|\n");
}

void printNetwork(struct Network n)
{
	printf("%d layers\n",n.nbLayers);
	printf("Number of neurons :\n");
	int *begin = n.nbNeurons;
	int *end = n.nbNeurons + n.nbLayers;
	array_print(begin, end);
	for (int i = 0; i < n.nbLayers; i++){
		printf("Layer %d (", i);
		printf("%d neurons):\n", n.layers[i].nbNeurons);
	  for (int j = 0; j < n.layers[i].nbNeurons; j++){
	   printf("Neuron %d: ", j);
	   printf("bias = %f\n", n.layers[i].neurons[j].bias);
	     for(int k = 0; k < n.layers[i].neurons[j].nbInputs;k++)
	       printf("weight = %f\n", n.layers[i].neurons[j].weights[k]);
	  }
	}
}

int main(int argc, char *argv[])
{ // hard-coded neural network for XOR function
	// 3 layers
		// 2 inputs
		// 2 neurons in the hidden layer
		// 1 output
	printf("Initiating program\n");

	struct Network network;
	int n0,n1,n2;
	if (argc == 4){
		n0 = strtoul(argv[1], NULL, 10);
		n1 = strtoul(argv[2], NULL, 10);
		n2 = strtoul(argv[3], NULL, 10);
	}
	else{
		n0 = 2;
		n1 = 2;
		n2 = 1;
	}
	int _nbNeurons[] = {n0,n1,n2};
	initNetwork(&network, 3, _nbNeurons);

// hard-coded bias and weights
	network.layers[1].neurons[0].bias = -10;
	network.layers[1].neurons[1].bias = 30;
	network.layers[2].neurons[0].bias = -30;
	network.layers[1].neurons[0].weights[0] = 20;
	network.layers[1].neurons[0].weights[1] = 20;
	network.layers[1].neurons[1].weights[0] = -20;
	network.layers[1].neurons[1].weights[1] = -20;
	network.layers[2].neurons[0].weights[0] = 20;
	network.layers[2].neurons[0].weights[1] = 20;

	printNetwork(network);

	float _testInputs[] = {1.0,0.0};
	float _testInputs2[] = {0.0,0.0};
	float *res = calloc(1, sizeof(float));
	float *res2 = calloc(1, sizeof(float));
	res = feedforward(network, 1, _testInputs);
	res2 = feedforward(network, 1, _testInputs2);
	printf("%f XOR %f\n", _testInputs[0], _testInputs[1]);
	printf("= %f\n", res[0]);
	printf("%f XOR %f\n", _testInputs2[0], _testInputs2[1]);
	printf("= %f\n", res2[0]);

	/*size_t lenTest = 1;
	float **testInputsList = malloc(lenTest * sizeof(float*));
	int *testOutputs = malloc(lenTest * sizeof(int));
	testInputsList[0] = _testInputs;
	testOutputs[0] = 0;
	printf("evaluate : %d\n",
		 evaluate(network, testInputs, testOutputs, lenTest));*/
	printf("Result (index output) = %d\n", test(network, _testInputs));
	printf("End\n");
	return 0;
}
