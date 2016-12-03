
## Simple SDL mini code

CC= gcc

CPPFLAGS= `pkg-config --cflags sdl`
CFLAGS= -Wall -Wextra -Werror -std=c99 -g3
LDFLAGS=
LDLIBS= `pkg-config --libs sdl` -lSDL_image


SRC= pixel_operations.c main.c create_array.c
OBJ= ${SRC:.c=.o}

all: main
	gcc -Wall -Wextra -std=c99 -g  network.c buildDB.c \
	main.c create_array.c pixel_operations.c \
	-o networkTest -lm \
	${LDLIBS} ${CPPFLAGS} #marche pas car fonction main deja definie

main: ${OBJ}

clean:
	rm -f *~ *.o
	rm -f main
	rm -f networkTest

cleantmp:
	rm -f tmp_*

fullclean: clean cleantmp

# END
