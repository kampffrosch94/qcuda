CC=gcc
CFLAGS=-std=c11 -Wall -march=native -g -O0
BIN=./bin
EXE=$(BIN)/qbitmachine
SRC=qbitmachine.c

PARAM=./input/15.pre 15

$(EXE): $(SRC)
	mkdir -p $(BIN)
	$(CC) -o $@ $^ $(CFLAGS)

valgrind:
	valgrind --leak-check=yes $(EXE) $(PARAM) 

run: $(EXE)
	$(EXE) $(PARAM)

time: $(EXE)
	time $(EXE) $(PARAM)

clean:
	rm $(BIN)/*

gdb:
	gdb $(EXE)

.PHONY: valgrind run clean gdb time
