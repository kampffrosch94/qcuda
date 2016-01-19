CC=gcc
CFLAGS=-std=c11 -Wall -O2 -march=native
BIN=./bin
EXE=$(BIN)/qbitmachine.c
SRC=qbitmachine.c

PARAM=10

$(EXE): $(SRC)
	$(CC) -o $@ $^ $(CFLAGS)

valgrind:
	valgrind --leak-check=yes $(EXE) $(PARAM) 

run:
	$(EXE) $(PARAM)

.PHONY: valgrind run
