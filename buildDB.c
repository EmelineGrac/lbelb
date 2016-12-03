#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <err.h>

#include "pixel_operations.h"
#include "create_array.h"
#include "main.h"
#include "network.h"

#include "buildDB.h"

//find . -name "*.db" -type f -delete
void buildDatabaseFileFromImg()
{
  FILE *fileTD = fopen(DATABASE, "wb");
  size_t size_td = 10009; //ls -lR | grep ".gif" | wc -l
  size_t size_inputs = 20*20;
  size_t size_outputs = 'z' - 'A';

  struct TrainingData *td = malloc(size_td * sizeof (struct TrainingData));
  size_t t = 0;

  char *path = calloc(255, sizeof (char));
 // {'t','r','a','i','n','i','n','g','/','c','/'};
  char c = 'A';
  int z = 0;
  FILE *f = NULL;
  while (c <= 'z')
  {
    if (c == 'Z' + 1)
      c = 'a';
    SDL_Surface *img = NULL;
    // path[9] = c;
    for (z = 0; z <= 201; ++z)
    {
      sprintf(path, "training/%d/%d.gif", c, z);
      // path[11] = z;
      f = fopen(path, "rb");
     if (f != NULL)
     {
      img = load_image(path);
      int *arr = makeArray(img);
      float *arrf = calloc(20 * 20, sizeof (float));
      for (unsigned k = 0; k < 400; k++)
        arrf[k] = (float)arr[k];
      SDL_FreeSurface(img);
      free(arr);
      td[t].trainingInputs = arrf;
      td[t].res = c - 'A';
      td[t].desiredOutput = indexOutputToVector(td[t].res, size_outputs);
      t++;
      fclose(f);
     }
    }
    c++;
  }
  free(path);
  buildDataBase(fileTD, td, size_td, size_inputs, size_outputs);
  fclose(fileTD);
  freeMemoryTD(&td, size_td);
  free(td);
}
