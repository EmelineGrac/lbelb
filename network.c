# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include "network.h"

float sigmoid(float z){
	return 1.0 / (1.0 + exp(-z));
}

float sigmoid_prime(float z){
	return sigmoid(z)*(1-sigmoid(z));
}

float cost_derivative(float output_activation, int y){
	return (output_activation - y);
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

void backprop(struct Network* n, float* trainingInputs, int desiredOutput){
/*
Update delta_nabla_bw
*/

int i,j,k;
for (i = 0; i < n->nbLayers; i++){
    for (j = 0; j < n->layers[i].nbNeurons; j++){
      n->layers[i].neurons[j].delta_nabla_b = 0;
      for(k = 0; k < n->layers[i].neurons[j].nbInputs;k++){
        n->layers[i].neurons[j].delta_nabla_w[k] = 0;
      }
    }
  }
for(j = 0; j < n->layers[0].nbNeurons; j++)
	// input layer activations are training inputs
	n->layers[0].neurons[j].activation = trainingInputs[j];
struct Neuron *nr; // must use a pointer!!!!!!
struct Layer l,ll;
float delta;
float sp;
for (i = 1; i < n->nbLayers; i++){
  for (j = 0; j < n->layers[i].nbNeurons; j++){
	nr = &(n->layers[i].neurons[j]);
	nr->z = 0;
	for(k = 0; k < nr->nbInputs; k++){
		nr->z += n->layers[i-1].neurons[k].activation
			 * nr->weights[k];
	}
	nr->z += nr->bias;
	nr->activation = sigmoid(nr->z);
  }
}
// backward pass
l = n->layers[n->nbLayers - 1];
ll= n->layers[n->nbLayers - 2];
for(j = 0; j < l.nbNeurons; j++){
  nr = &(l.neurons[j]);
  delta = cost_derivative(nr->activation,desiredOutput)
			 * sigmoid_prime(nr->z);
  nr->delta_nabla_b = delta;
  for(k = 0; k < nr->nbInputs; k++){
	nr->delta_nabla_w[k] = ll.neurons[k].activation
			 * nr->delta_nabla_b;
  }
}

for(i = n->nbLayers - 2; i > 0; i--){
 l = n->layers[i];
 ll = n->layers[i-1];
   for(j = 0; j < l.nbNeurons; j++){
     nr = &(l.neurons[j]);
     sp = sigmoid_prime(nr->z);
     delta = 0;
     for(k = 0; k < n->layers[i+1].nbNeurons; k++){
	delta += n->layers[i+1].neurons[k].weights[j]
		 * n->layers[i+1].neurons[k].delta_nabla_b;
     }
     delta *= sp;
     nr->delta_nabla_b = delta;
     for(k = 0; k < nr->nbInputs; k++){
	nr->delta_nabla_w[k] = ll.neurons[k].activation
			 * nr->delta_nabla_b;
     }
  }
}
}

void update_mini_batch(struct Network* n, struct TrainingData* k,
struct TrainingData* k_end, float eta){

/*
Update the network's weights and biases
by applying gradient descent using backpropagation
to a single mini batch.
The mini_batch is an array of struct TrainingData
eta is the learning rate.
 */

 int i, j, kk;
 size_t len = k_end - k;

  for (i = 0; i < n->nbLayers; i++){
    for (j = 0; j < n->layers[i].nbNeurons; j++){
      n->layers[i].neurons[j].nabla_b = 0;
      n->layers[i].neurons[j].delta_nabla_b = 0;
      for(kk = 0; kk < n->layers[i].neurons[j].nbInputs;kk++){
        n->layers[i].neurons[j].nabla_w[kk] = 0;
        n->layers[i].neurons[j].delta_nabla_w[kk] = 0;
      }
    }
  }

for(; k < k_end; k++){
  backprop(n,(*k).trainingInputs,(*k).desiredOutput);

  for (i = 0; i < n->nbLayers; i++){
    for (j = 0; j < n->layers[i].nbNeurons; j++){
      n->layers[i].neurons[j].nabla_b
	 += n->layers[i].neurons[j].delta_nabla_b;
      for(kk = 0; kk < n->layers[i].neurons[j].nbInputs;kk++)
        n->layers[i].neurons[j].nabla_w[kk]
	 += n->layers[i].neurons[j].delta_nabla_w[kk];
    }
  }
}

struct Neuron* nr;

  for (i = 0; i < n->nbLayers; i++){
    for (j = 0; j < n->layers[i].nbNeurons; j++){
      nr = &(n->layers[i].neurons[j]);
      nr->bias = nr->bias - (eta/len) * (nr->nabla_b);
      for(kk = 0; kk < n->layers[i].neurons[j].nbInputs;kk++)
        nr->weights[kk] = nr->weights[kk] - (eta/len) * (nr->nabla_w[kk]);
    }
  }
}

void SGD(struct Network* n, struct TrainingData* td,
	 size_t size_td, int epochs, int mini_batch_size,
	 float eta){
/*
td is a list of struct TrainingData(trainingInputs[],desiredOutput)
One epoch consists of one full training cycle on the training set.
Once every sample in the set is seen, you start again,
marking the beginning of the 2nd epoch.
mini_batch_size is the size of one sample.
eta is the learning rate.
*/
  for(int j = 0; j < epochs; j++){
	// random.shuffle(td);
	struct TrainingData* k = td;
	struct TrainingData* k_end = td + size_td;
	for(; k < k_end; k += mini_batch_size)
		update_mini_batch(n,k,k+mini_batch_size,eta);
  }
}


void initNeuron(struct Neuron* _neuron, float _bias, int _nbInputs)
{
	_neuron->bias = _bias;
	_neuron->nbInputs = _nbInputs;
	float *_weights = malloc(_nbInputs * sizeof(float));
	float *_nabla_w = malloc(_nbInputs * sizeof(float));
	float *_delta_nabla_w = malloc(_nbInputs * sizeof(float));
	float rn;
	float rn_max = 1;
	for(int i = 0; i < _nbInputs; i++){
		// rn will be the same
		rn = -0.5+(float)rand()/(float)(RAND_MAX/rn_max);
		_weights[i] = rn;
	}
	_neuron->weights = _weights;
	_neuron->nabla_b = 0.0;
	_neuron->delta_nabla_b = 0.0;
	_neuron->nabla_w = _nabla_w;
	_neuron->delta_nabla_w = _delta_nabla_w;
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
	   //printf("nabla_b = %f\n", n.layers[i].neurons[j].nabla_b);
	     for(int k = 0; k < n.layers[i].neurons[j].nbInputs;k++){
	       printf("weight = %f\n", n.layers[i].neurons[j].weights[k]);
	     //printf("nabla_w = %f\n", n.layers[i].neurons[j].nabla_w[k]);
	     }
          }
	}
}

int main(int argc, char *argv[])
{ // hard-coded neural network for XOR function
	// 3 layers
		// 2 inputs
		// 2 neurons in the hidden layer
		// 1 output
	printf("Random biases and weights:\n");
	srand(time(NULL));
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
		n2 = 1; // TODO: 2 outputs;
	}
	int _nbNeurons[] = {n0,n1,n2};
	initNetwork(&network, 3, _nbNeurons);

// hard-coded bias and weights
/*	network.layers[1].neurons[0].bias = -10;
	network.layers[1].neurons[1].bias = 30;
	network.layers[2].neurons[0].bias = -30;
	network.layers[1].neurons[0].weights[0] = 20;
	network.layers[1].neurons[0].weights[1] = 20;
	network.layers[1].neurons[1].weights[0] = -20;
	network.layers[1].neurons[1].weights[1] = -20;
	network.layers[2].neurons[0].weights[0] = 20;
	network.layers[2].neurons[0].weights[1] = 20;
*/
	printNetwork(network);

	float _testInputs00[] = {0.0,0.0};
	float _testInputs01[] = {0.00,1.0};
	float _testInputs10[] = {1.00,0.00};
	float _testInputs11[] = {1.00,1.00};

	/*size_t lenTest = 1;
	float **testInputsList = malloc(lenTest * sizeof(float*));
	int *testOutputs = malloc(lenTest * sizeof(int));
	testInputsList[0] = _testInputs;
	testOutputs[0] = 0;
	printf("evaluate : %d\n",
		 evaluate(network, testInputs, testOutputs, lenTest));*/
//	printf("Result (index output) = %d\n", test(network, _testInputs00));

	printf("Test SGD :\n");

	size_t size_td = 4;
	struct TrainingData* td =
	 malloc(size_td * sizeof(struct TrainingData));

	struct TrainingData td1;
	td1.trainingInputs = _testInputs00;
	td1.desiredOutput = 0;

	struct TrainingData td2;
	td2.trainingInputs = _testInputs01;
	td2.desiredOutput = 1;

	struct TrainingData td3;
	td3.trainingInputs = _testInputs10;
	td3.desiredOutput = 1;

	struct TrainingData td4;
	td4.trainingInputs = _testInputs11;
	td4.desiredOutput = 0;

	td[0] = td1;
	td[1] = td2;
	td[2] = td3;
	td[3] = td4;

	int epochs = 1000000;
	int mini_batch_size = 2;
	float eta = 4.0;
	printf(" Size of TrainingData = %zu\n %d epochs\n \
mini_batch_size = %d\n eta = %f\n...",
	size_td, epochs, mini_batch_size, eta);
	SGD(&network, td, size_td, epochs, mini_batch_size, eta);

	printNetwork(network);

	float *res = calloc(1, sizeof(float));// TODO : 2
	float *res2 = calloc(1, sizeof(float));
	res = feedforward(network, 1, _testInputs00);
	res2 = feedforward(network, 1, _testInputs01);
	float *res3 = calloc(1, sizeof(float));
	float *res4 = calloc(1, sizeof(float));
	res3 = feedforward(network, 1, _testInputs10);
	res4 = feedforward(network, 1, _testInputs11);

	printf("feedforward...\n");
	printf("%f XOR %f\n", _testInputs00[0], _testInputs00[1]);
	printf("= %f\n", res[0]);
	printf("%f XOR %f\n", _testInputs01[0], _testInputs01[1]);
	printf("= %f\n", res2[0]);
	printf("%f XOR %f\n", _testInputs10[0], _testInputs10[1]);
	printf("= %f\n", res3[0]);
	printf("%f XOR %f\n", _testInputs11[0], _testInputs11[1]);
	printf("= %f\n", res4[0]);

	printf("End\n");
	return 0;
}
