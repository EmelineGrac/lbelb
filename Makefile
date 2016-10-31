
## Simple SDL mini code
 
CC= gcc
 
CPPFLAGS= `pkg-config --cflags sdl`
CFLAGS= -Wall -Wextra -Werror -std=c99 -O3 -g3
LDFLAGS=
LDLIBS= `pkg-config --libs sdl` -lSDL_image
 
SRC= pixel_operations.c main.c
OBJ= ${SRC:.c=.o}
 
all: main
	gcc -Wall -Wextra -std=c99  network.c -o networkTest -lm
 
main: ${OBJ}
 
clean:
	rm -f *~ *.o
	rm -f main
	rm networkTest
 
# END
