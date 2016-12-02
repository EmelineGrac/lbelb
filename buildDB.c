#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include "pixel_operations.h"
#include <err.h>
#include "create_array.h"
#include "main.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "network.h"

void build()
{
  FILE *fileTD = fopen("testData.bin", "wb");
  size_t size_td = 4; //nb of images
  size_t size_inputs = 20*20;
  size_t size_outputs = 2;

  struct TrainingData *td = malloc(size_td * sizeof (struct TrainingData));
  size_t t = 0;

  char path[] =
  {'t','r','a','i','n','i','n','g','/','c','/','z','.','g','i','f','\0'};
  char c = 'A';
  char z = '0';

  while (c <= 'B')
  {
    if (c == 'Z' + 1)
      c = 'a';
    SDL_Surface *img = malloc(sizeof (SDL_Surface));
    path[9] = c;
    for (z = '0'; z <= '1'; ++z)
    {
      path[11] = z;
      img = load_image(path);
      int *arr = calloc(20 * 20, sizeof (int));
      float *arrf = calloc(20 * 20, sizeof (float));
      arr = makeArray(img);
      for (unsigned k = 0; k < 400; k++)
      {
        arrf[k] = (float)arr[k];
      }
      free(img);
      free(arr);
      td[t].trainingInputs = arrf;
      td[t].res = c - 'A';
      td[t].desiredOutput = indexOutputToVector(td[t].res, size_outputs);
      t++;
    }
    c++;
  }
  buildDataBase(fileTD, td, size_td, size_inputs, size_outputs);
  fclose(fileTD);
}
