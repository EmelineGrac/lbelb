// create_array.h
 
# ifndef CREATE_ARRAY_H_
# define CREATE_ARRAY_H_
 
# include <stdlib.h>
# include <SDL/SDL.h> 
# include <SDL.h>
# include <SDL/SDL_image.h>                                                     
# include <err.h>                                                               
# include "pixel_operations.h"                                                  
 
int* makeArray(/*char *argv [],*/ SDL_Surface *surface);
void segmentation(int* array/*, char *argv []*/, SDL_Surface *surface);

# endif
