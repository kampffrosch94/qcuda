CC=gcc
CFLAGS=-std=c11 -Wall -O2 -march=native
BIN=./bin
EXE=$(BIN)/qbitmachine.c
SRC=qbitmachine.c

PARAM=10

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
