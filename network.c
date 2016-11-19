#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "network.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

inline float sigmoid(float z)
{
	return 1.0 / (1.0 + exp(-z));
}

inline float sigmoid_prime(float z)
{
	return sigmoid(z) * (1 - sigmoid(z));
}

inline float cost_derivative(float output_activation, int y)
{
	return output_activation - y;
}

/*
** Softmax function for the last layer
*/
inline float softmax(struct Network *n, int j)
{
	struct Layer *l = &(n->layers[n->nbLayers - 1]);
	return exp(l->neurons[j].z) / l->sum_outputs;
}

/*
** Generate a random number following a Gaussian distribution
** with mean 0 and standard deviation 1.
** The Box-Muller transform is used.
*/
double random_normal(void)
{
	return sqrt(-2 * log((rand() + 1.0) / (RAND_MAX + 1.0)))
	* cos(2 * PI * (rand() + 1.0) / (RAND_MAX + 1.0));
}


/*
** Return the index of the output which has the highest score
*/
int highest(float *result, int size)
{
	int res = 0;
	float max = result[0];

	for (int i = 0; i < size; i++)
	{
		if (result[i] > max)
		{
			max = result[i];
			res = i;
		}
	}
	return res;
}


/*
** return the output of the network if inputsVect is input
** input layer = 0
*/
float* feedforward(struct Network *n, int iLayer, float *inputsVect)
{
	if (iLayer == n->nbLayers)
		return inputsVect;
	else
	{
		int j = 0;
		float* outputsVect = calloc(n->layers[iLayer].nbNeurons,
					    sizeof (float));
		struct Neuron *nr = &(n->layers[iLayer].neurons[0]);
		n->layers[iLayer].sum_outputs = 0.0;
		while (j < n->layers[iLayer].nbNeurons)
		{
			nr = &(n->layers[iLayer].neurons[j]);
			nr->z = 0;
			for (int k = 0; k < nr->nbInputs; k++)
				nr->z += inputsVect[k] * nr->weights[k];
			nr->z += nr->bias;
			if (iLayer < n->nbLayers - 1)
				outputsVect[j] = sigmoid(nr->z);
			else
				n->layers[iLayer].sum_outputs += exp(nr->z);
			j++;
		}
		if (iLayer == n->nbLayers - 1)
		{
			for (j = 0; j < n->layers[iLayer].nbNeurons; j++)
				outputsVect[j] = softmax(n, j);
		}
		if (iLayer > 1)
			free(inputsVect);
		// The function is at first called with iLayer = 1.
		// In the first call inputsVect is not allocated,
		// so no need to free it.
		return feedforward(n, iLayer + 1, outputsVect);
	}
}

int test(struct Network *n, float *inputsVect)
{
  int res;
  float *activations;
  activations = feedforward(n, 1, inputsVect);
  res = highest(activations, n->layers[n->nbLayers-1].nbNeurons);
  free(activations);
  return res;
}


/*
** Return the number of test inputs for which the neural
** network outputs the correct result. Note that the neural
** network's output is assumed to be the index of whichever
** neuron in the final layer has the highest activation.
*/
int evaluate(struct Network *n, struct TrainingData td[], size_t size_td)
{
 int res = 0;
 float *activations;

 for (size_t i = 0; i < size_td; i++)
 {
   activations = feedforward(n, 1, td[i].trainingInputs);
   if (highest(activations, n->layers[n->nbLayers-1].nbNeurons) == td[i].res)
     res++;
   free(activations);
 }
 return res;
}


/*
** Backpropagation algorithm
** Update delta_nabla_bw
*/
void backprop(struct Network *n, float *trainingInputs,	int* desiredOutput)
{
 int i, j, k;

// 0.Initialization to ZERO

 for (i = 0; i < n->nbLayers; i++)
 {
    for (j = 0; j < n->layers[i].nbNeurons; j++)
    {
      n->layers[i].neurons[j].delta_nabla_b = 0;
      for (k = 0; k < n->layers[i].neurons[j].nbInputs; k++)
        n->layers[i].neurons[j].delta_nabla_w[k] = 0;
    }
  }

// 1. Set the corresponding activation for the input layer.
// Input layer activations are training inputs

 for (j = 0; j < n->layers[0].nbNeurons; j++)
   n->layers[0].neurons[j].activation = trainingInputs[j];

// 2. Feedforward: For each l=2,3,…,L
// compute zl=wlal−1+bl and al=sigmoid(zl).

 struct Neuron *nr;
 struct Layer l, ll;
 float delta;
 float sp;

 for (i = 1; i < n->nbLayers - 1; i++)
 {
   for (j = 0; j < n->layers[i].nbNeurons; j++)
   {
	 nr = &(n->layers[i].neurons[j]);
	 nr->z = 0;

	 for (k = 0; k < nr->nbInputs; k++)
       nr->z += n->layers[i - 1].neurons[k].activation * nr->weights[k];
	 nr->z += nr->bias;

	 nr->activation = sigmoid(nr->z);
   }
 }

  // last layer use the softmax function instead of the sigmoid one
  n->layers[n->nbLayers - 1].sum_outputs = 0.0;

  // compute the denominator: sum over all the output neurons
  for (j = 0; j < n->layers[n->nbLayers - 1].nbNeurons; j++)
  {
    nr = &(n->layers[n->nbLayers - 1].neurons[j]);
    nr->z = 0;

    for (k = 0; k < nr->nbInputs; k++)
      nr->z += n->layers[i - 1].neurons[k].activation * nr->weights[k];
    nr->z += nr->bias;

    n->layers[n->nbLayers - 1].sum_outputs += exp(nr->z);
  }

  // compute the activations
  for (j = 0; j < n->layers[n->nbLayers - 1].nbNeurons; j++)
  {
    nr = &(n->layers[n->nbLayers - 1].neurons[j]);
    nr->activation = softmax(n, j);
  }

// backward pass

// 3. Output error: Compute the vector δL = ∇aC ⊙ σ′(zL).

  l  = n->layers[n->nbLayers - 1];
  ll = n->layers[n->nbLayers - 2];

  for (j = 0; j < l.nbNeurons; j++)
  {
    nr = &(l.neurons[j]);
    delta = cost_derivative(nr->activation, desiredOutput[j])
            * sigmoid_prime(nr->z);
    nr->delta_nabla_b = delta;
    for (k = 0; k < nr->nbInputs; k++)
      nr->delta_nabla_w[k] = ll.neurons[k].activation * nr->delta_nabla_b;
  }

// 4. Backpropagate the error: For each l=L−1,L−2,…,2 compute
// δl=((wl+1)δl+1) ⊙  σ′(zl).

 for(i = n->nbLayers - 2; i > 0; i--)
 {
   l  = n->layers[i];
   ll = n->layers[i - 1];

   for (j = 0; j < l.nbNeurons; j++)
   {
     nr = &(l.neurons[j]);
     sp = sigmoid_prime(nr->z);
     delta = 0;

	 for (k = 0; k < n->layers[i + 1].nbNeurons; k++)
	   delta +=   n->layers[i + 1].neurons[k].weights[j]
		        * n->layers[i + 1].neurons[k].delta_nabla_b;
     delta *= sp;

// 5. Output: Compute the gradient of the cost function.

     nr->delta_nabla_b = delta;
     for (k = 0; k < nr->nbInputs; k++)
	   nr->delta_nabla_w[k] = ll.neurons[k].activation
		                      * nr->delta_nabla_b;
  }
 }
}


/*
** Update the network's weights and biases
** by applying gradient descent using backpropagation
** to a single mini batch.
** The mini_batch is an array of struct TrainingData
** eta is the learning rate.
*/
void update_mini_batch(struct Network *n,
                       struct TrainingData *k,
                       struct TrainingData *k_end,
                       float eta)
{
 int i, j, kk;
 size_t len = k_end - k;
 struct Neuron *nr;

 for (i = 0; i < n->nbLayers; i++)
 {
    for (j = 0; j < n->layers[i].nbNeurons; j++)
    {
      nr = &(n->layers[i].neurons[j]);
      nr->nabla_b = 0;
      nr->delta_nabla_b = 0;
      for (kk = 0; kk < nr->nbInputs; kk++)
      {
        nr->nabla_w[kk] = 0;
        nr->delta_nabla_w[kk] = 0;
      }
    }
 }

 for(; k < k_end; k++)
 {
    backprop(n, k->trainingInputs, k->desiredOutput);

    for (i = 0; i < n->nbLayers; i++)
    {
      for (j = 0; j < n->layers[i].nbNeurons; j++)
      {
        nr = &(n->layers[i].neurons[j]);
        nr->nabla_b += nr->delta_nabla_b;

        for (kk = 0; kk < nr->nbInputs;kk++)
          nr->nabla_w[kk] += nr->delta_nabla_w[kk];
      }
    }
 }

 for (i = 0; i < n->nbLayers; i++)
 {
    for (j = 0; j < n->layers[i].nbNeurons; j++)
    {
      nr = &(n->layers[i].neurons[j]);
      nr->bias -= ((eta/len) * (nr->nabla_b));
      for (kk = 0; kk < n->layers[i].neurons[j].nbInputs;kk++)
        nr->weights[kk] -= ((eta/len) * (nr->nabla_w[kk]));
    }
  }
}


/*
** td is a list of struct TrainingData(trainingInputs[],desiredOutput)
** One epoch consists of one full training cycle on the training set.
** Once every sample in the set is seen, you start again,
** marking the beginning of the 2nd epoch.
** mini_batch_size is the size of one sample.
** eta is the learning rate.
*/
void SGD(struct Network *n,
         struct TrainingData *td,
         size_t size_td,
         int epochs,
         int mini_batch_size,
         float eta)
{
  struct TrainingData *k = td;
  struct TrainingData *k_end = td + size_td;

  for (int j = 0; j < epochs; j++)
  {
	// random.shuffle(td);
	k = td;
	for (; k < k_end; k += mini_batch_size)
		update_mini_batch(n, k, k + mini_batch_size, eta);
  }
}


void initNeuron(struct Neuron *_neuron, float _bias, int _nbInputs)
{
	float *_weights = malloc(_nbInputs * sizeof (float));
	float *_nabla_w = malloc(_nbInputs * sizeof (float));
	float *_delta_nabla_w = malloc(_nbInputs * sizeof (float));
	float rn;

	for (int i = 0; i < _nbInputs; i++)
	{
		rn = random_normal() / sqrt(_nbInputs);
		_weights[i] = rn;
	}

	_neuron->bias = _bias;
	_neuron->nbInputs = _nbInputs;
	_neuron->weights = _weights;
	_neuron->nabla_b = 0.0;
	_neuron->delta_nabla_b = 0.0;
	_neuron->nabla_w = _nabla_w;
	_neuron->delta_nabla_w = _delta_nabla_w;
}

void initLayer(struct Layer *_layer, int _nbNeurons, int _nbInputs)
{
	struct Neuron *_neurons = malloc(_nbNeurons * sizeof (struct Neuron));
	float rn = 0.0;

	for (int i = 0; i < _nbNeurons; i++)
	{
		rn = random_normal();
		initNeuron(&_neurons[i], rn, _nbInputs);
	}

	_layer->neurons = _neurons;
	_layer->nbNeurons = _nbNeurons;
	_layer->sum_outputs = 0.0;
}

void initNetwork(struct Network *_network, int _nbLayers, int *_nbNeurons)
{
	struct Layer *_layers = malloc(_nbLayers * sizeof (struct Layer));
	int _nbInputs = 0; // first layer has no weights

	for (int i = 0; i < _nbLayers; i++)
	{
		initLayer(&_layers[i], _nbNeurons[i],_nbInputs);
		_nbInputs = _layers[i].nbNeurons;
	}

	_network->layers = _layers;
	_network->nbLayers = _nbLayers;
	_network->nbNeurons = _nbNeurons;
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

void printNetwork(struct Network *n)
{
	printf("\nNeural network with %d layers\n", n->nbLayers);
	printf("Number of neurons:  ");
	int *begin = n->nbNeurons;
	int *end = n->nbNeurons + n->nbLayers;
	array_print(begin, end);
	for (int i = 0; i < n->nbLayers; i++)
        {
		printf("Layer %d (", i);
		printf("%d neurons):\n", n->layers[i].nbNeurons);
	  for (int j = 0; j < n->layers[i].nbNeurons; j++)
          {
	   printf("     Neuron %d: ", j);
	   printf("bias = %f\n", n->layers[i].neurons[j].bias);
	     for (int k = 0; k < n->layers[i].neurons[j].nbInputs; k++)
             {
	       printf("               weight = %f\n",
			 n->layers[i].neurons[j].weights[k]);
	     }
          }
	}
}

void openWeightsFile(struct Network *n, char fileName[])
{
	FILE* f = fopen(fileName, "r");
	int i, j, k, ll;
	struct Neuron *nr;

	int _nbLayers = 0;
	fscanf(f, "%d\n", &_nbLayers);

	int *_nbNeurons = calloc(sizeof (int), _nbLayers);
	int *begin = _nbNeurons;
	int *end = _nbNeurons + _nbLayers;

	for (; begin < end - 1; ++begin)
		fscanf(f, "%d ", begin);
	fscanf(f, "%d\n", end - 1);
	initNetwork(n, _nbLayers, _nbNeurons);

	for (i = 0; i < n->nbLayers; i++)
	{
	  for (j = 0; j < n->layers[i].nbNeurons; j++)
	  {
	    nr = &(n->layers[i].neurons[j]);
	    fscanf(f, "%f ", &(nr->bias));
	    if (nr->nbInputs > 0)
	    {
	       ll = nr->nbInputs - 1;
	       for (k = 0; k < ll; k++)
	          fscanf(f, "%f ", &(nr->weights[k]));
               fscanf(f, "%f\n", &(nr->weights[ll]));
            }
            else
               fscanf(f, "\n");

	  }
	}
	fclose(f);
}

void writeWeightsFile(struct Network *n, char fileName[])
{
	FILE* f = fopen(fileName, "w");
	int i, j, k, ll;
	struct Neuron nr;

	fprintf(f, "%d\n",n->nbLayers);
	int *begin = n->nbNeurons;
	int *end = n->nbNeurons + n->nbLayers;
	for (; begin < end - 1; ++begin)
		fprintf(f, "%d ", *begin);
	fprintf(f, "%d\n", *(end - 1));

	for (i = 0; i < n->nbLayers; i++)
	{
	  for (j = 0; j < n->layers[i].nbNeurons; j++)
	  {
	    nr = n->layers[i].neurons[j];
	    fprintf(f, "%f ", nr.bias);
	    if (nr.nbInputs > 0)
	    {
	      ll = nr.nbInputs - 1;
	      for (k = 0; k < ll; k++)
	          fprintf(f, "%f ", nr.weights[k]);
              fprintf(f, "%f\n", nr.weights[ll]);
	    }
	    else
              fprintf(f, "\n");
          }
	}
	fclose(f);
}


void freeMemoryNetwork(struct Network *n)
{
	// free(n->layers); will cause invalid read
	for (int j = 0; j < n->nbLayers; j++)
	{
		for (int k = 0; k < n->layers[j].nbNeurons; k++)
		{
			free(n->layers[j].neurons[k].weights);
			free(n->layers[j].neurons[k].nabla_w);
			free(n->layers[j].neurons[k].delta_nabla_w);
		}
		free(n->layers[j].neurons);
        }
	free(n->layers);
	free(n->nbNeurons);
}

void randomInit(struct Network *n)
{
	int n1, n2, n3;

	printf("\nNumber of neurons on layer 1: ");
	scanf("%d", &n1);

	printf("\nNumber of neurons on layer 2: ");
	scanf("%d", &n2);

	printf("\nNumber of neurons on layer 3: ");
	scanf("%d", &n3);

	int *_nbNeurons = malloc(3 * sizeof (int));
	*_nbNeurons = n1;
	*(_nbNeurons + 1) = n2;
	*(_nbNeurons + 2) = n3;
	printf("\nRandom biases and weights:\n");
	initNetwork(n, 3, _nbNeurons);
}

int* indexOutputToVector(int index, size_t len)
{
	int *res = calloc(len, sizeof (int));
	res[index] = 1;
	return res;
}

// BUILD TEXT FILE FUNCTIONS
int isAcceptedByNeuralNetwork(float *input)
{
  // USELESS CURRENTLY
  // IF NOT A SINGLE CHARACTER (LIKE \N)
  // RETURN 0
  // ELSE
  // RETURN 1
  if (*input)
    return 1;
  return 1;
}

int specialTreatment(float *input)
{
  // USELESS CURRENTLY
  // CONVERT SPECIAL INPUT TO CHAR
  if (*input)
    return 1;
  return '\n';
}

int outputInt2Char(int outputInt)
{
  // CONVERT TO ASCII CODE
  int c_res = outputInt + 48;
  return c_res;
}


void buildResultFile(struct Network *n,
                     float **inputs,
                     size_t len,
                     char *fileName)
{
  FILE* f = fopen(fileName, "w");
  size_t i = 0;
  int res = 0;
  int c_res = 0;
  for (; i < len; i++)
  {
     if (!isAcceptedByNeuralNetwork(inputs[i]))
     {
        c_res = specialTreatment(inputs[i]);
     }
     else
     {
        res = test(n, inputs[i]);
        c_res = outputInt2Char(res);
     }
     fputc(c_res, f);
  }
  fclose(f);
}


int main()
{
// Loading neural network
	srand(time(NULL));
	struct Network *network = malloc(sizeof (struct Network));

	char mode[50];
	char fileName[50];

	printf("Mode: loadWeightsFile, new\n");
	scanf("%s", mode);

	if (strcmp(mode,"loadWeightsFile") == 0)
	{
		printf("fileName: ");
		scanf("%s", fileName);
		openWeightsFile(network, fileName);
	}
	else
		randomInit(network);
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
	//could be a static array of size 4, use stack instead of heap
	 malloc(size_td * sizeof(struct TrainingData));

	struct TrainingData td1;
	td1.trainingInputs = _testInputs00;
	td1.desiredOutput = r00;
	td1.res = 0;

	struct TrainingData td2;
	td2.trainingInputs = _testInputs01;
	td2.desiredOutput = r01;
	td2.res = 1;

	struct TrainingData td3;
	td3.trainingInputs = _testInputs10;
	td3.desiredOutput = r10;
	td3.res = 1;

	struct TrainingData td4;
	td4.trainingInputs = _testInputs11;
	td4.desiredOutput = r11;
	td4.res = 0;

	td[0] = td1;
	td[1] = td2;
	td[2] = td3;
	td[3] = td4;

	int expectedOutputs[] = {0, 1, 1, 0};
	float **evaluationInputs = malloc(4 * sizeof(float *));
	evaluationInputs[0] = _testInputs00;
	evaluationInputs[1] = _testInputs01;
	evaluationInputs[2] = _testInputs10;
	evaluationInputs[3] = _testInputs11;

	float *res;
	float *res2;
	float *res3;
	float *res4; // = calloc(2, sizeof(float));
	// memory already allocated in feedforward


// First evaluation
	evalres = evaluate(network, td, 4);
	printf("\nEvaluation : %d / 4 --> ", evalres);

	if (evalres != 4)
		printf("FAIL\n");
	else
		printf("SUCCESS\n");

	do {
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

// Training and evaluation (loop)

	SGD(network, td, size_td, epochs, mini_batch_size, eta);
	evalres = evaluate(network, td, 4);

	printNetwork(network);
	printf("\nEvaluation : %d / 4 --> ", evalres);

	res = feedforward(network, 1, _testInputs00);
	res2 = feedforward(network, 1, _testInputs01);
	res3 = feedforward(network, 1, _testInputs10);
	res4 = feedforward(network, 1, _testInputs11);


	if (evalres != 4)
	{
	printf("FAIL\n");
	printf("\nTests results (FAILED):\n");
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

	freeMemoryNetwork(network);
	randomInit(network);
	printNetwork(network);
	free(res); // will be reallocated when feedforward will be called
	free(res2);
	free(res3);
	free(res4);
	}
	else
		printf("SUCCESS\n");
	} while (evalres != 4);

// Tests results

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

	printf("Print results in file 'results'\n");
	buildResultFile(network, evaluationInputs, 4, "results");
//Save
	if (evalres == 4)
	{
		printf("Write weights file? no/fileName\n");
		scanf("%s", fileName);
		if (strcmp("no", fileName) != 0)
			writeWeightsFile(network, fileName);
	}

//Free memory
	free(td);
	free(res);
	free(res2);
	free(res3);
	free(res4);
	free(evaluationInputs);

	freeMemoryNetwork(network);
	free(network);
	printf("End\n");
	return 0;
}
