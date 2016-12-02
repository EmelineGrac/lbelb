
## Simple SDL mini code

CC= gcc

CPPFLAGS= `pkg-config --cflags sdl`
CFLAGS= -Wall -Wextra -Werror -std=c99 -g3
LDFLAGS=
LDLIBS= `pkg-config --libs sdl` -lSDL_image -lm


SRC= pixel_operations.c main.c create_array.c buildDB.c network.c
# j'ai renomme la fonction main de main.c, desole
OBJ= ${SRC:.c=.o}

all: main
#	gcc -Wall -Wextra -std=c99 -g pixel_operations.c main.c create_array.c \
#	buildDB.c network.c -lSDL_image -lm
# marche pas, merde

main: ${OBJ}

clean:
	rm -f *~ *.o
	rm -f main

cleantmp:
	rm -f tmp_*

fullclean: clean cleantmp

# END
