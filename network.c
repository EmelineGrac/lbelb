# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>
# include "network.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

float sigmoid(float z){
	return 1.0 / (1.0 + exp(-z));
}

float sigmoid_prime(float z){
	return sigmoid(z) * (1 - sigmoid(z));
}

float cost_derivative(float output_activation, int y){
	return output_activation - y;
}

double random_normal(void){
/* Generate a random number following a Gaussian distribution
** with mean 0 and standard deviation 1.
** The Box-Muller transform is used.
*/
	return sqrt(-2 * log((rand() + 1.0) / (RAND_MAX + 1.0)))
	* cos(2 * PI * (rand() + 1.0) / (RAND_MAX + 1.0));
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
		float* outputsVect = calloc(n.layers[iLayer].nbNeurons,
						sizeof(float));
		struct Neuron nr = n.layers[iLayer].neurons[0];
		while (j < n.layers[iLayer].nbNeurons){
			res = 0;
			nr = n.layers[iLayer].neurons[j];
			for(int k = 0; k < nr.nbInputs; k++){
				res += inputsVect[k] * nr.weights[k];}

			outputsVect[j] = sigmoid(res + nr.bias);
			j++;
		}
		if (iLayer > 1)
			free(inputsVect);
		// The function is at first called with iLayer = 1.
		// In the first call inputsVect is not allocated,
		// so no need to free it.
		return feedforward(n, iLayer + 1, outputsVect);
	}
}

int test(struct Network n, float* inputsVect){
	return highest(feedforward(n, 1, inputsVect),
			n.layers[n.nbLayers-1].nbNeurons);
}

int evaluate(struct Network n, float **inputs,
		 int *outputs, size_t len){
/* Return the number of test inputs for which the neural
** network outputs the correct result. Note that the neural
** network's output is assumed to be the index of whichever
** neuron in the final layer has the highest activation.
*/
	int res = 0;
	float *activations;
	for(size_t i = 0; i < len; i++){
		activations = feedforward(n, 1, inputs[i]);
		if (highest(activations,
			n.layers[n.nbLayers-1].nbNeurons)
				 == outputs[i])
			res++;
		free(activations);
	}
	return res;
}

void backprop(struct Network* n,
		float* trainingInputs,
		int* desiredOutput){
/* Update delta_nabla_bw
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
struct Neuron *nr; // must use a pointer
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
  delta = cost_derivative(nr->activation,desiredOutput[j])
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

void update_mini_batch(struct Network* n,
		struct TrainingData* k,
		struct TrainingData* k_end,
		float eta){

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
	//?Address 0x_ is 0 bytes after a block of size 0 alloc'd

	float *_nabla_w = malloc(_nbInputs * sizeof(float));
	float *_delta_nabla_w = malloc(_nbInputs * sizeof(float));
	float rn;
	for(int i = 0; i < _nbInputs; i++){
		rn = random_normal() / sqrt(_nbInputs);
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
	float rn = 0.0;
	for(int i = 0; i < _nbNeurons; i++){
		rn = random_normal();
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
	int _nbInputs = 0;// first layer has no weights
	for(int i = 0; i < _nbLayers; i++){
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
    line += printf("| %2d ", *begin);
  }
  printf("|\n");
}

void printNetwork(struct Network n)
{
	printf("\nNeural network with %d layers\n",n.nbLayers);
	printf("Number of neurons:  ");
	int *begin = n.nbNeurons;
	int *end = n.nbNeurons + n.nbLayers;
	array_print(begin, end);
	for (int i = 0; i < n.nbLayers; i++){
		printf("Layer %d (", i);
		printf("%d neurons):\n", n.layers[i].nbNeurons);
	  for (int j = 0; j < n.layers[i].nbNeurons; j++){
	   printf("     Neuron %d: ", j);
	   printf("bias = %f\n", n.layers[i].neurons[j].bias);
	   //printf("nabla_b = %f\n", n.layers[i].neurons[j].nabla_b);
	     for(int k = 0; k < n.layers[i].neurons[j].nbInputs;k++){
	       printf("               weight = %f\n",
			 n.layers[i].neurons[j].weights[k]);
	     //printf("nabla_w = %f\n", n.layers[i].neurons[j].nabla_w[k]);
	     }
          }
	}
}

void open(struct Network *n, char fileName[])
{
	FILE* f = fopen(fileName, "r");
	int i,j,k,ll;

	int _nbLayers = 0;
	fscanf(f, "%d\n", &_nbLayers);

	int *_nbNeurons = calloc(sizeof(int), _nbLayers);
	int *begin = _nbNeurons;
	int *end = _nbNeurons + _nbLayers;
	for(;begin < end - 1; ++begin)
		fscanf(f, "%d ", begin);
	fscanf(f, "%d\n", end - 1);
	initNetwork(n, _nbLayers, _nbNeurons);

	for (i = 0; i < n->nbLayers; i++){
	  for (j = 0; j < n->layers[i].nbNeurons; j++){
	   fscanf(f, "%f ", &(n->layers[i].neurons[j].bias));
	   ll = (n->layers[i].neurons[j].nbInputs) - 1;
//TODO BUG
	     for(k = 0; k < ll;k++)
	       fscanf(f, "%f ", &(n->layers[i].neurons[j].weights[k]));
	     fscanf(f, "%f\n",&(n->layers[i].neurons[j].weights[k]));
//The code above causes Invalid write of size 4 for the last weight

/*No problem with the code below... ?!
             for(k = 0; k < ll + 1; k++)
	       fscanf(f, "%f ", &(n->layers[i].neurons[j].weights[k]));
*/
          }
	}
	fclose(f);
}

void write(struct Network n, char fileName[])
{
	FILE* f = fopen(fileName, "w");
	int i,j,k,ll;

	fprintf(f, "%d\n",n.nbLayers);
	int *begin = n.nbNeurons;
	int *end = n.nbNeurons + n.nbLayers;
	for(;begin < end - 1; ++begin)
		fprintf(f, "%d ", *begin);
	fprintf(f, "%d\n", *(end - 1));

	for (i = 0; i < n.nbLayers; i++){
	  for (j = 0; j < n.layers[i].nbNeurons; j++){
	   fprintf(f, "%f ", n.layers[i].neurons[j].bias);
	   ll = (n.layers[i].neurons[j].nbInputs) - 1;
//TODO BUG
	     for(k = 0; k < ll;k++)
	       fprintf(f, "%f ", n.layers[i].neurons[j].weights[k]);
	     fprintf(f, "%f\n",n.layers[i].neurons[j].weights[k]);
//The code above causes Invalid read of size 4 for the last weight

/*No problem with the code below... ?!
	     for(k = 0; k < ll + 1; k++)
	       fprintf(f, "%f ", n.layers[i].neurons[j].weights[k]);
*/
          }
	}
	fclose(f);
}


void freeMemoryNetwork(struct Network* n)
{
	// free(n->layers); will cause invalid read
	for (int j = 0; j < n->nbLayers; j++){
		// free(n->layers[j].neurons);
		for(int k = 0; k < n->layers[j].nbNeurons;k++){
			free(n->layers[j].neurons[k].weights);
			free(n->layers[j].neurons[k].nabla_w);
			free(n->layers[j].neurons[k].delta_nabla_w);
		}
		free(n->layers[j].neurons);
        }
	free(n->layers);
	free(n->nbNeurons);//if nbNeurons is a dynamic array
}

void randomInit(struct Network *n){

	int n1, n2, n3;

	printf("\nNumber of neurons on layer 1: ");
	scanf("%d", &n1);

	printf("\nNumber of neurons on layer 2: ");
	scanf("%d", &n2);

	printf("\nNumber of neurons on layer 3: ");
	scanf("%d", &n3);

	int *_nbNeurons = malloc(3 * sizeof(int));
	*_nbNeurons = n1;
	*(_nbNeurons + 1) = n2;
	*(_nbNeurons + 2) = n3;
	printf("\nRandom biases and weights:\n");
	initNetwork(n, 3, _nbNeurons);
}

int* indexOutputToVector(int index, size_t len){
		int *res = calloc(len, sizeof(int));
		res[index] = 1;
		return res;
	}


int main()
{
// Loading neural network
	srand(time(NULL));
	struct Network network;

	char mode[50];
	char fileName[50];

	printf("Mode: loadFile, new\n");
	scanf("%s", mode);

	if (strcmp(mode,"loadFile") == 0){
		printf("fileName: ");
		scanf("%s", fileName);
		open(&network, fileName);
	}
	else
		randomInit(&network);
	printNetwork(network);


// Training

	int evalres = 0;

	size_t size_td = 4;
	int epochs = 10000;
	int mini_batch_size = 2;
	float eta = 4.0;

	float _testInputs00[] = {0.0,0.0};
	float _testInputs01[] = {0.00,1.0};
	float _testInputs10[] = {1.00,0.00};
	float _testInputs11[] = {1.00,1.00};

	int r00[] = {1,0};
	int r01[] = {0,1};
	int r10[] = {0,1};
	int r11[] = {1,0};

	struct TrainingData* td =
	 malloc(size_td * sizeof(struct TrainingData));

	struct TrainingData td1;
	td1.trainingInputs = _testInputs00;
	td1.desiredOutput = r00;

	struct TrainingData td2;
	td2.trainingInputs = _testInputs01;
	td2.desiredOutput = r01;

	struct TrainingData td3;
	td3.trainingInputs = _testInputs10;
	td3.desiredOutput = r10;

	struct TrainingData td4;
	td4.trainingInputs = _testInputs11;
	td4.desiredOutput = r11;

	td[0] = td1;
	td[1] = td2;
	td[2] = td3;
	td[3] = td4;

	int expectedOutputs[] = {0, 1, 1, 0};
	float **evaluationInputs = malloc(8 * sizeof(float));
	evaluationInputs[0] = _testInputs00;
	evaluationInputs[1] = _testInputs01;
	evaluationInputs[2] = _testInputs10;
	evaluationInputs[3] = _testInputs11;
	evalres = evaluate(network,evaluationInputs,
				expectedOutputs,4);
	printf("\nEvaluation : %d / 4 --> ", evalres);
	if (evalres != 4)
		printf("FAIL\n");
	else
		printf("SUCCESS\n");

	do{

	printf("\nUse SGD for training (XOR)\n");
	printf("\nepochs: ");
	scanf("%d", &epochs);
	printf("\nmini_batch_size: ");
	scanf("%d", &mini_batch_size);
	printf("\neta: ");
	scanf("%f", &eta);

	printf("\n  Size of TrainingData = %zu\n  epochs = %d\n \
 mini_batch_size = %d\n  eta = %f\n\n",
	size_td, epochs, mini_batch_size, eta);

	SGD(&network, td, size_td, epochs, mini_batch_size, eta);

// Evaluation


	evalres = evaluate(network,evaluationInputs,
				expectedOutputs,4);

	printNetwork(network);

	printf("\nEvaluation : %d / 4 --> ", evalres);
	if (evalres != 4){
		printf("FAIL\n");
		freeMemoryNetwork(&network);
		randomInit(&network);
		printNetwork(network);
	}
	else
		printf("SUCCESS\n");
	} while (evalres != 4);

// Tests results

	float *res;
	float *res2;
	float *res3;
	float *res4; // = calloc(2, sizeof(float));
	// memory already allocated in feedforward
	res = feedforward(network, 1, _testInputs00);
	res2 = feedforward(network, 1, _testInputs01);
	res3 = feedforward(network, 1, _testInputs10);
	res4 = feedforward(network, 1, _testInputs11);

	printf("\nTests results:\n");
	printf("%.0f XOR %.0f\n", _testInputs00[0], _testInputs00[1]);
	printf("= %f %f\n", res[0],res[1]);
	printf("%d\n\n", highest(res,2));
	printf("%.0f XOR %.0f\n", _testInputs01[0], _testInputs01[1]);
	printf("= %f %f\n", res2[0],res2[1]);
	printf("%d\n\n", highest(res2,2));
	printf("%.0f XOR %.0f\n", _testInputs10[0], _testInputs10[1]);
	printf("= %f %f\n", res3[0],res3[1]);
	printf("%d\n\n", highest(res3,2));
	printf("%.0f XOR %.0f\n", _testInputs11[0], _testInputs11[1]);
	printf("= %f %f\n", res4[0],res4[1]);
	printf("%d\n\n", highest(res4,2));


//Save
	if (evalres == 4){
		printf("Write file? no/fileName\n");
		scanf("%s", fileName);
		if (strcmp("no", fileName) != 0)
			write(network, fileName);
	}

//Free memory
	free(td);
	free(res);
	free(res2);
	free(res3);
	free(res4);
	free(evaluationInputs);

	freeMemoryNetwork(&network);

	printf("End\n");
	return 0;
}
