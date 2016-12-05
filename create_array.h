// create_array.h

# ifndef CREATE_ARRAY_H_
# define CREATE_ARRAY_H_

# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <SDL/SDL.h>
# include <SDL.h>
# include <SDL/SDL_image.h>
# include <err.h>
# include "pixel_operations.h"

int* makeArray(SDL_Surface *surface);
int* makeArrayW1B0(SDL_Surface *surface);
int** segmentation(int* array, SDL_Surface *surface, int* len);
int* tabLetter(int *array);
# endif
