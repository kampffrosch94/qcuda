CC=gcc
CFLAGS=-std=c11 -Wall -march=native -g
BIN=./bin
EXE=$(BIN)/qbitmachine
SRC=qbitmachine.c

PARAM=./input/10.pre 10

$(EXE): $(SRC)
	mkdir -p $(BIN)
	$(CC) -o $@ $^ $(CFLAGS)

valgrind:
	valgrind --leak-check=yes $(EXE) $(PARAM) 

run:
	$(EXE) $(PARAM)

clean:
	rm $(BIN)/*

.PHONY: valgrind run clean
