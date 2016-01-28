N=16
CC=nvcc
CFLAGS=-O3 #-D "Ndef ${N}"#--compiler-options "-std=c11 -Wall -march=native -O2" #-pg
BIN=./bin
EXE=$(BIN)/qbitmachine
SRC=qbitmachine.cu

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
