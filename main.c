#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <SDL/SDL.h>
#include <SDL.h>
#include <SDL/SDL_image.h>
#include <err.h>

#include "pixel_operations.h"
#include "buildDB.h"
#include "imag.h"
#include "create_array.h"
#include "network.h"

int main(int argc, char *argv[])
{
  if (argc >= 1)
  {
    unsigned n = strtoul(argv[1], NULL, 10);
    if (n == 3)
      learning();
    if (argc >= 2)
    {
      char *path = argv[2];
      if (n == 1)
        main_imag(path);
      if (n == 2)
        OCR(path);
    }
  }
  return 0;
}
