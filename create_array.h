// create_array.h
 
# ifndef CREATE_ARRAY_H_
# define CREATE_ARRAY_H_
 
# include <stdlib.h>
# include <SDL/SDL.h> 
# include <SDL.h>
# include <SDL/SDL_image.h>                                                     
# include <err.h>                                                               
# include "pixel_operations.h"                                                  

/*struct tab {
  int		color;
  int		colom;
  int		ligne;
  struct tab	*next;
}*/


int* makeArray(/*char *argv [],*/ SDL_Surface *surface);
int** segmentation(int* array/*, char *argv []*/, SDL_Surface *surface);
SDL_Surface* NouvelleImage(int *array, SDL_Surface* img);

# endif
