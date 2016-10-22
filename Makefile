networkTest:	
	gcc -Wall -Wextra -std=c99 -g network.c -o networkTest -lm

clean:
	rm networkTest
