## Simple SDL mini code

CC= gcc

CPPFLAGS= `pkg-config --cflags sdl`
CFLAGS= -Wall -Wextra -Werror -std=c99 -g3
LDFLAGS=
LDLIBS= `pkg-config --libs sdl` -lSDL_image -lm


SRC= pixel_operations.c imag.c create_array.c network.c buildDB.c
OBJ= ${SRC:.c=.o}

all: main

main: ${OBJ}

clean:
	rm -f *~ *.o
	rm -f main
	rm -f networkTest

cleantmp:
	rm -f tmp_*

fullclean: clean cleantmp

# END
