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
  unsigned arg = strtoul(argv[1], NULL, 10);
  if (arg == 1)
    main_imag(argc, argv);
  if (arg == 2)
    OCR(argc, argv);
  if (arg == 3)
    learning();
  return 0;
}
