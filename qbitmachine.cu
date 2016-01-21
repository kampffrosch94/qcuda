#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
#define CUDA_CHECK_KERNEL {cudaError_t error = cudaGetLastError(); if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}


__global__ void count_combinations(int64_t *preset, uint64_t *found_combinations){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;

  int64_t bv = preset[4 * idx];
  int64_t bh = preset[4 * idx + 1];
  int64_t bu = preset[4 * idx + 2];
  int64_t bd = preset[4 * idx + 3];
  
  //TODO Input Algo
  
}



int main(int argc, char ** argv){
    if(argc < 2){
        printf("Not enough arguments\n");
        return 1;
    }
    int32_t N = atoi(argv[1]);
	
	
	int blocksize=1024;
	dim3 dimBlock(blocksize);
	dim3 dimGrid(1);
	
	//TODO
	int size = blocksize;
	
	uint64_t * found_combinations_host = malloc(blocksize,sizeof(uint64_t));
	for(int i = 0; i < blocksize; i++){
		found_combinations_host[i] = 0;
	}
	
	
	int64_t preset_host[blocksize*4];
	
	CUDA_CHECK(cudaMalloc((void**)&preset_device,size*4*sizeof(int64_t)));
	CUDA_CHECK(cudaMalloc((void**)&found_combinations_device,size*sizeof(uint64_t)));
	
	CUDA_CHECK(cudaMemcpy(preset_device,preset_host,size*4*sizeof(int64_t),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(found_combinations_device,found_combinations_host,size*sizeof(uint64_t),cudaMemcpyHostToDevice));
	
	
	//TODO
	count_combinations<<<dimGrid,dimBlock>>>(preset_device,found_combinations_device);
	CUDA_CHECK_KERNEL
	
	
	
	CUDA_CHECK(cudaMemcpy(found_combinations_host,found_combinations_device,size*sizeof(uint64_t),cudaMemcpyDeviceToHost));
	
	uint_64 found_combinations = 0;
	for(int i = 0; i < size; i++){
		found_combinations += found_combinations_host[i];
	}
	printf("\nFound Combinations: %" PRIu64 "\n", found_combinations);
	
	
	CUDA_CHECK(cudaFree(found_combinations_device));
	CUDA_CHECK(cudaFree(preset_device));
	
}