N=27
CC=nvcc
CFLAGS=-O3 -arch=compute_35 -code=sm_35
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
