CC=gcc
CFLAGS=-std=c11 -Wall -march=native -O0 -pg
BIN=./bin
EXE=$(BIN)/qbitmachine
SRC=qbitmachine.c

PARAM=./input/15.pre 15
#PARAM=./input/10.pre 10

$(EXE): $(SRC)
	mkdir -p $(BIN)
	$(CC) -o $@ $^ $(CFLAGS)

valgrind: $(EXE)
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
