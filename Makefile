CC=gcc
CFLAGS=-std=c11 -Wall -march=native -O2 #-pg
BIN=./bin
EXE=$(BIN)/qbitmachine
SRC=qbitmachine.c

N=15
PARAM=./input/$(N).pre $(N)

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
