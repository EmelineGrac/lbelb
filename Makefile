all:
	gcc -Wall -Wextra -std=c99  network.c -o networkTest -lm



clean:
	rm networkTest
