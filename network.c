#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "buildDB.h"
#include "network.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef WEIGHTS_FILE
#define WEIGHTS_FILE "tmp_weights.txt"
#endif

#ifndef RESULTS_FILE
#define RESULTS_FILE "tmp_results.txt"
#endif

#ifndef DATABASE
#define DATABASE "testData.bin"
#endif

#ifndef XOR
#define XOR 0 // 1 = specific demo for XOR
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
int highest(float result[], int size)
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
float *feedforward(struct Network *n,
                   int             iLayer,
                   float           inputsVect[])
{
    if (iLayer == n->nbLayers)
        return inputsVect;
    else
    {
        int j = 0;
        float *outputsVect = calloc(n->layers[iLayer].nbNeurons,
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

int test(struct Network *n, float inputsVect[])
{
  int res;
  float *activations = NULL;
  activations = feedforward(n, 1, inputsVect);
  res = highest(activations, n->layers[n->nbLayers - 1].nbNeurons);
  free(activations);
  return res;
}


/*
** Return the number of test inputs for which the neural
** network outputs the correct result. Note that the neural
** network's output is assumed to be the index of whichever
** neuron in the final layer has the highest activation.
*/
size_t evaluate(struct Network      *n,
                struct TrainingData  td[],
                size_t               size_td)
{
 size_t res = 0;
 float *activations = NULL;
 int high = 0;

 for (size_t i = 0; i < size_td; i++)
 {
   activations = feedforward(n, 1, td[i].trainingInputs);
   high = highest(activations, n->layers[n->nbLayers - 1].nbNeurons);
   if (high == td[i].res)
     res++;
   else
   {
     printf("%c was supposed to be %c\n",
     outputIntToChar(high), outputIntToChar(td[i].res));
   }
   free(activations);
 }
 return res;
}


/*
** Backpropagation algorithm
** Update delta_nabla_bw
*/
void backprop(struct Network *n,
              float           trainingInputs[],
              int             desiredOutput[])
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
void update_mini_batch(struct Network      *n,
                       struct TrainingData *td,
                       struct TrainingData *td_end,
                       float                eta)
{
 int i, j, k;
 size_t len = td_end - td;
 struct Neuron *nr;

 for (i = 0; i < n->nbLayers; i++)
 {
    for (j = 0; j < n->layers[i].nbNeurons; j++)
    {
      nr = &(n->layers[i].neurons[j]);
      nr->nabla_b = 0;
      nr->delta_nabla_b = 0;
      for (k = 0; k < nr->nbInputs; k++)
      {
        nr->nabla_w[k] = 0;
        nr->delta_nabla_w[k] = 0;
      }
    }
 }

 for(; td < td_end; td++)
 {
    backprop(n, td->trainingInputs, td->desiredOutput);

    for (i = 0; i < n->nbLayers; i++)
    {
      for (j = 0; j < n->layers[i].nbNeurons; j++)
      {
        nr = &(n->layers[i].neurons[j]);
        nr->nabla_b += nr->delta_nabla_b;

        for (k = 0; k < nr->nbInputs; k++)
          nr->nabla_w[k] += nr->delta_nabla_w[k];
      }
    }
 }

 for (i = 0; i < n->nbLayers; i++)
 {
    for (j = 0; j < n->layers[i].nbNeurons; j++)
    {
      nr = &(n->layers[i].neurons[j]);
      nr->bias -= ((eta/len) * (nr->nabla_b));
      for (k = 0; k < n->layers[i].neurons[j].nbInputs; k++)
        nr->weights[k] -= ((eta/len) * (nr->nabla_w[k]));
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
void SGD(struct Network      *n,
         struct TrainingData  td[],
         size_t               size_td,
         unsigned             epochs,
         unsigned             mini_batch_size,
         float                eta)
{
  if (mini_batch_size == 0)
    mini_batch_size = 1;

  struct TrainingData *begin = td;
  struct TrainingData *end   = td + size_td;

  for (unsigned j = 0; j < epochs; j++)
  {
    // random.shuffle(td);
    begin = td;
    for (; begin < end; begin += mini_batch_size)
    {
        if (begin + mini_batch_size <= end)
            update_mini_batch(n, begin, begin + mini_batch_size, eta);
        else
            update_mini_batch(n, begin, end, eta);
    }
  }
}

/*
** Same but with evaluation for each epoch
*/
void SGD_eval(struct Network      *n,
              struct TrainingData  td[],
              size_t               size_td,
              unsigned             epochs,
              unsigned             mini_batch_size,
              float                eta)
{
  if (mini_batch_size == 0)
    mini_batch_size = 1;

  struct TrainingData *begin = td;
  struct TrainingData *end   = td + size_td;

  for (unsigned j = 0; j < epochs; j++)
  {
    // random.shuffle(td);
    begin = td;
    for (; begin < end; begin += mini_batch_size)
    {
        if (begin + mini_batch_size <= end)
            update_mini_batch(n, begin, begin + mini_batch_size, eta);
        else
            update_mini_batch(n, begin, end, eta);
    }
    unsigned evalres = evaluate(n, td, size_td);
    printf("Epoch %u: %u / %zu\n", j, evalres, size_td);
  }
}

void initNeuron(struct Neuron *_neuron, float _bias, int _nbInputs)
{
    float *_weights       = malloc(_nbInputs * sizeof (float));
    float *_nabla_w       = malloc(_nbInputs * sizeof (float));
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
        initLayer(&_layers[i], _nbNeurons[i], _nbInputs);
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


/*
** Build a binary file that stores trainingData.
*/
void buildDataBase(FILE                *f,
                   struct TrainingData  td[],
                   size_t               size_td,
                   size_t               size_inputs,
                   size_t               size_outputs)
{
// Write sizes
    fwrite(&(size_td),      sizeof (size_t), 1, f);
    fwrite(&(size_inputs),  sizeof (size_t), 1, f);
    fwrite(&(size_outputs), sizeof (size_t), 1, f);

// Write data
    struct TrainingData *begin = td;
    struct TrainingData *end   = td + size_td;
    for (; begin < end; ++begin)
    {
        fwrite(begin->trainingInputs, sizeof (float), size_inputs,  f);
        fwrite(&(begin->res),         sizeof (int),   1,            f);
        fwrite(begin->desiredOutput,  sizeof (int),   size_outputs, f);
    }
}

/*
** Build an array of trainingData from a binary file.
*/
void readDataBase(FILE                 *f,
                  struct TrainingData  *td[], // ref ptr
                  size_t               *size_td,
                  size_t               *size_inputs,
                  size_t               *size_outputs)
{
// Read sizes
    fread((size_td),      sizeof (size_t), 1, f);
    fread((size_inputs),  sizeof (size_t), 1, f);
    fread((size_outputs), sizeof (size_t), 1, f);

// Allocate
    *td = malloc(sizeof (struct TrainingData) * (*size_td));
    struct TrainingData *begin = *td;
    struct TrainingData *end   = *td + *size_td;
    for (; begin < end; ++begin)
    {
        begin->trainingInputs = malloc(sizeof (float) * (*size_inputs));
        begin->desiredOutput  = malloc(sizeof (int)   * (*size_outputs));
        fread(begin->trainingInputs, sizeof (float), *size_inputs,  f);
        fread(&(begin->res),         sizeof (int),    1,            f);
        fread(begin->desiredOutput,  sizeof (int),   *size_outputs, f);
    }
}


void freeMemoryNetwork(struct Network *n)
{
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

void freeMemoryTD(struct TrainingData *td[], size_t size_td)
{
    struct TrainingData *begin = *td;
    struct TrainingData *end   = *td + size_td;
    for (; begin < end; ++begin)
    {
        free(begin->trainingInputs);
        free(begin->desiredOutput);
    }
}

void randomInitScanf(struct Network *n)
{
    int n1, n2, n3;

    printf("Number of neurons on layer 1: ");
    scanf("%d", &n1);

    printf("Number of neurons on layer 2: ");
    scanf("%d", &n2);

    printf("Number of neurons on layer 3: ");
    scanf("%d", &n3);

    int *_nbNeurons = malloc(3 * sizeof (int));
    *_nbNeurons = n1;
    *(_nbNeurons + 1) = n2;
    *(_nbNeurons + 2) = n3;
    printf("Random biases and weights\n");
    initNetwork(n, 3, _nbNeurons);
}

void randomInit(struct Network *n, int input, int hidden, int output)
{
    int *_nbNeurons = malloc(3 * sizeof (int));
    *_nbNeurons = input;
    *(_nbNeurons + 1) = hidden;
    *(_nbNeurons + 2) = output;
    initNetwork(n, 3, _nbNeurons);
}

int* indexOutputToVector(int index, size_t len)
{
    int *res = calloc(len, sizeof (int));
    if (index < (int)len)
        res[index] = 1;
    return res;
}

int outputVectorToIndex(int outputs[], int size)
{
    int i;
    for (i = 0; i < size && outputs[i] != 1; i++);
    return i;
}

// BUILD TEXT FILE FUNCTIONS
int isAcceptedByNeuralNetwork(float input[])
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

int specialTreatment(float input[])
{
  // USELESS CURRENTLY
  // CONVERT SPECIAL INPUT TO CHAR
  if (*input)
    return 1;
  return '\n';
}

int outputIntToChar(int outputInt)
{
  // CONVERT TO ASCII CODE
  int c_res = outputInt + 'A';
  return c_res;
}


void buildResultFile(struct Network *n,
                     int            *inputs[],
                     size_t          len,
                     char            fileName[])
{
  FILE* f = fopen(fileName, "w");
  size_t i = 0;
  int res = 0;
  int c_res = 0;
  float *arrf = NULL;

  for (; i < len; i++)
  {
     arrf = calloc(20 * 20, sizeof (float));
     for (unsigned k = 0; k < 400; k++)
        arrf[k] = (float)inputs[i][k];
     if (!isAcceptedByNeuralNetwork(arrf))
     {
        c_res = specialTreatment(arrf);
     }
     else
     {
        res = test(n, arrf);
        c_res = outputIntToChar(res);
     }
     fputc(c_res, f);
     free(arrf);
  }
  fputc('\n', f);
  fclose(f);
}

void buildResultFileTraining(struct Network      *n,
                             struct TrainingData  td[],
                             size_t               len,
                             char                 fileName[])
{
  FILE* f = fopen(fileName, "w");
  size_t i = 0;
  int res = 0;
  int c_res = 0;
  for (; i < len; i++)
  {
        res = test(n, td[i].trainingInputs);
        c_res = outputIntToChar(res);

     fputc(c_res, f);
  }
  fputc('\n', f);
  fclose(f);
}

int main()
{
// build database file
    buildDatabaseFileFromImg();
// Loading neural network, from a text file or randomly
    srand(time(NULL)); // for random
    struct Network *network = malloc(sizeof (struct Network));

// Database parameters
    size_t evalres = 0;
    FILE  *fileTD = NULL;
    size_t size_td = 4;
    size_t size_inputs = 2;
    size_t size_outputs = 2;
    struct TrainingData *td;
// Learning parameters
    int hidden = 2;
    unsigned epochs = 10000;
    unsigned mini_batch_size = 2;
    float eta = 4.0;

    int eval_during_training = 0; // 0 = FALSE

// Load trainingData from the binary file
    fileTD = fopen(DATABASE, "rb");
    td = NULL;
    readDataBase(fileTD, &td, &size_td, &size_inputs, &size_outputs);
    fclose(fileTD);

// Initialiaze the neural net
    char mode[50];
    char fileName[50];
    printf("Mode: loadWeightsFile(l), new(n) ");

    scanf("%s", mode);
    if (strcmp(mode,"l") == 0)
    {
        printf("fileName: ");
        scanf("%s", fileName);
        openWeightsFile(network, fileName);
        // in case of:
        if (network->nbNeurons[0] != (int)size_inputs) //bad
        {
            printf("Error on the number of neurons on the input layer: \
    network has %d inputs but size_inputs = %zu.\n", network->nbNeurons[0],
    size_inputs);
            return 1;
        }
        if (network->nbNeurons[2] != (int)size_outputs)
        {
            printf("Error on the number of neurons on the output layer: \
    network has %d outputs but size_outputs = %zu.\n", network->nbNeurons[2],
    size_outputs);
            return 1;
        }
    }
    else
    {
        printf("Number of neurons on the hidden layer: ");
        scanf("%d", &hidden);
        randomInit(network, size_inputs, hidden, size_outputs);
    }

// Define learning parameters
    printf("(SGD) Size of TrainingData: %zu\n", size_td);
    printf("epochs: ");
    scanf("%u", &epochs);
    printf("mini_batch_size: ");
    scanf("%u", &mini_batch_size);
    printf("eta: ");
    scanf("%f", &eta);

// use SGD for learning
    if (eval_during_training)
        SGD_eval(network, td, size_td, epochs, mini_batch_size, eta);
    else
        SGD(network, td, size_td, epochs, mini_batch_size, eta);

// weights visualization: printNetwork(network);

// Evaluation
    evalres = evaluate(network, td, size_td);
    printf("Evaluation : %zu / %zu\n", evalres, size_td);

// Save weights in a text file
    writeWeightsFile(network, WEIGHTS_FILE);
    printf("Print weights in file '%s'\n", WEIGHTS_FILE);
// Save results
    buildResultFileTraining(network, td, size_td, RESULTS_FILE);
    printf("Print results in file '%s'\n", RESULTS_FILE);

// Free memory
    freeMemoryTD(&td, size_td);
    free(td);
    freeMemoryNetwork(network);
    free(network);
    return 0;
}


int main_xor()
{
// Loading neural network, from a text file or randomly
    srand(time(NULL)); // for random
    struct Network *network = malloc(sizeof (struct Network));

    char mode[50];
    char fileName[50];

    printf("Mode: loadWeightsFile(l), new(n)\n");
    scanf("%s", mode);

    if (strcmp(mode,"l") == 0)
    {
        printf("fileName: ");
        scanf("%s", fileName);
        openWeightsFile(network, fileName);
    }
    else
        randomInitScanf(network);
#if XOR
    printNetwork(network);
#endif

// Training

    size_t evalres = 0;
    FILE  *fileTD = NULL;
    size_t size_td = 4;
    size_t size_inputs = 2;
    size_t size_outputs = 2;
    struct TrainingData *td;
// Learning parameters
    unsigned epochs = 10000;
    unsigned mini_batch_size = 2;
    float eta = 4.0;

#if XOR
// inputs for training
    float _testInputs00[] = {0.0,0.0};
    float _testInputs01[] = {0.00,1.0};
    float _testInputs10[] = {1.00,0.00};
    float _testInputs11[] = {1.00,1.00};

// build manually struct TrainingData, assign outputs
    td = malloc(size_td * sizeof (struct TrainingData));

    td[0].trainingInputs = _testInputs00;
    td[0].res = 0;
    td[0].desiredOutput = indexOutputToVector(td[0].res, 2);

    td[1].trainingInputs = _testInputs01;
    td[1].res = 1;
    td[1].desiredOutput = indexOutputToVector(td[1].res, 2);

    td[2].trainingInputs = _testInputs10;
    td[2].res = 1;
    td[2].desiredOutput = indexOutputToVector(td[2].res, 2);

    td[3].trainingInputs = _testInputs11;
    td[3].res = 0;
    td[3].desiredOutput = indexOutputToVector(td[3].res, 2);

// Save trainingData in a binary file
    fileTD = fopen("trainingData.bin", "wb");
    buildDataBase(fileTD, td, size_td, size_inputs, size_outputs);
    fclose(fileTD);

// Free memory, only desiredOutput has been malloc
    free(td[0].desiredOutput);
    free(td[1].desiredOutput);
    free(td[2].desiredOutput);
    free(td[3].desiredOutput);
    free(td);
#endif

// Load trainingData from the binary file
    fileTD = fopen("trainingData.bin", "rb");
    td = NULL;
    readDataBase(fileTD, &td, &size_td, &size_inputs, &size_outputs);
    fclose(fileTD);

#if XOR
    float *res;
    float *res2;
    float *res3;
    float *res4; // = calloc(2, sizeof(float));
    // memory already allocated in feedforward
#endif

#if XOR
// First evaluation: random
    evalres = evaluate(network, td, size_td);
    printf("\nEvaluation : %zu / %zu --> ", evalres, size_td);

    if (evalres != size_td)
        printf("FAIL\n");
    else
        printf("SUCCESS\n");
#endif

#if XOR
    do {
#endif
    printf("\nUse SGD for training (XOR)\n");
    printf("\nepochs: ");
    scanf("%u", &epochs);
    printf("\nmini_batch_size: ");
    scanf("%u", &mini_batch_size);
    printf("\neta: ");
    scanf("%f", &eta);

    printf("\n  Size of TrainingData = %zu\n  epochs = %d\n \
 mini_batch_size = %d\n  eta = %f\n\n",
    size_td, epochs, mini_batch_size, eta);

// Training and evaluation (loop)

// use SGD for learning
    SGD(network, td, size_td, epochs, mini_batch_size, eta);

    printNetwork(network);

// Evaluation
    evalres = evaluate(network, td, size_td);
    printf("\nEvaluation : %zu / %zu --> ", evalres, size_td);

#if XOR
    res = feedforward(network, 1, _testInputs00);
    res2 = feedforward(network, 1, _testInputs01);
    res3 = feedforward(network, 1, _testInputs10);
    res4 = feedforward(network, 1, _testInputs11);

    if (evalres != 4) // FAIL
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
    else // SUCCESS
        printf("SUCCESS\n");
    } while (evalres != 4);

// Print tests results in the shell

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

// Print results in a text file
    printf("Print results in file 'results'\n");

    float **evaluationInputs = malloc(4 * sizeof(float *));
    evaluationInputs[0] = _testInputs00;
    evaluationInputs[1] = _testInputs01;
    evaluationInputs[2] = _testInputs10;
    evaluationInputs[3] = _testInputs11;
    buildResultFile(network, evaluationInputs, 4, "results");
#endif
// Save weights in a text file
    if (evalres == size_td)
    {
        printf("\nWrite weights file? no/fileName\n");
        scanf("%s", fileName);
        if (strcmp("no", fileName) != 0)
            writeWeightsFile(network, fileName);
    }

// Free memory
    freeMemoryTD(&td, size_td);
    free(td);
#if XOR
    free(res);
    free(res2);
    free(res3);
    free(res4);
    free(evaluationInputs);
#endif
    freeMemoryNetwork(network);
    free(network);
    printf("End\n");
    return 0;
}
