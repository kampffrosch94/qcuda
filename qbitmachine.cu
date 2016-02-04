//#define _BSD_SOURCE 
#define Ndef 27
#define Ldef 2
#define Blocksizedef 1
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <endian.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
#define CUDA_CHECK_KERNEL {cudaError_t error = cudaGetLastError(); if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

typedef struct {
    uint64_t bh;
    uint64_t bu;
    uint64_t bd;
    uint64_t bv;
    uint64_t result;
    uint8_t sym;
} cuda_pre;

typedef struct {
    uint64_t m_spec;
    uint64_t m_sol;
} preplacement;

__global__ void count_combinations(cuda_pre * pre,const uint32_t pre_count){//3 < N < 64

    int id=blockIdx.x*blockDim.x+threadIdx.x;
    int idx = threadIdx.x;
    if(id < pre_count){
        const uint32_t N = Ndef;
        const uint32_t L = Ldef;
        uint64_t bv = pre[id].bv; //block vertical has to be preset
        uint64_t bh = pre[id].bh; //block horizontal; count from bottom to top
        uint64_t bu = pre[id].bu; //element 0 on the lower left
        uint64_t bd = pre[id].bd; //element 0 on the lower right

        __shared__ uint64_t cols[Blocksizedef][N];
        //the field on which the queens are set, split in columns
        //onehot coded for the row in which the queen is placed
        //0 means unset
        __shared__ int64_t free_vectors[Blocksizedef][N];
        //has 1 in positions which are free in the
        //current col (opposite to bh)
        for(int32_t i = 0; i < N; i++){
            cols[idx][i] = 0;
            free_vectors[idx][i] = 0;
        }

        uint64_t relevant_bits = (1 << N) - 1;

        uint64_t found_combinations = 0;

        int32_t start = L;
        int32_t end = N - 1 - L;

        while(((bv >> start)&1)&&(start < end)){
            start++;
        }

        while(((bv >> end)&1)&&(end > start)){
            end--;
        }


        int32_t cc = start; //current column
    
        free_vectors[idx][cc] = ~(bh | (bu >> cc) | (bd >> (N -1 - cc))) 
            & relevant_bits;

        while(1){
            if(cols[idx][cc] > 0){
                //delete the old placement in the blockvectors (if any)
                bh ^= cols[idx][cc];
                bu ^= cols[idx][cc] << cc;
                bd ^= cols[idx][cc] << (N - 1 - cc);
            }

            if(free_vectors[idx][cc] == 0){ //track back
                if(cc == start){ //stop
                    break;
                }
                cols[idx][cc] = 0;
                cc--;
                while(((bv >> cc)&1)){
                    cc--;
                }
            } else {
                //find the next new placement in the current col
                cols[idx][cc] = free_vectors[idx][cc] & -free_vectors[idx][cc];
                //remove it from the free_vector
                free_vectors[idx][cc] ^= cols[idx][cc];
                //set the new placement in the blockvectors
                bh ^= cols[idx][cc];
                bu ^= cols[idx][cc] << cc;
                bd ^= cols[idx][cc] << (N - 1 - cc);

                if(cc == end){ 
                    found_combinations++;
                } else {
                    cc++;
                    while(((bv >> cc)&1)){
                        cc++;
                    }
                    free_vectors[idx][cc] = ~(bh | (bu >> cc) | (bd >> (N - 1 - cc))) 
                        & relevant_bits;
                }
            }
        }
        pre[id].result = found_combinations;
    }
}

void set_position(int32_t x,int32_t y, int32_t N,uint64_t * bh, 
        uint64_t * bu, uint64_t * bd, uint64_t * bv){
    uint64_t tbh = 1 << y;
    uint64_t tbu = tbh << x;
    uint64_t tbd = tbh << (N - 1 - x);
    uint64_t tbv = 1 << x;

    *bh |= tbh;
    *bu |= tbu;
    *bd |= tbd;
    *bv |= tbv;

#if (DEBUG > 1)
    printf("(%d,%d)\n",x,y);
#endif
}

cuda_pre evaluate_preplacement(int32_t N, uint64_t spec){
    uint8_t wa,wb,na,nb,ea,eb,sa,sb,sym;

    wa = (spec >> 60) & 0b1111; 
    wb = (spec >> 55) & 0b11111;
    na = (spec >> 50) & 0b11111;
    nb = (spec >> 45) & 0b11111;
    ea = (spec >> 40) & 0b11111;
    eb = (spec >> 35) & 0b11111;
    sa = (spec >> 30) & 0b11111;
    sb = (spec >> 25) & 0b11111;
    sym = (spec >> 23) & 0b11;

    uint64_t bh = 0;
    uint64_t bu = 0;
    uint64_t bd = 0;
    uint64_t bv = 0;

    set_position(0,wa,N,&bh,&bu,&bd,&bv);
    set_position(1,wb,N,&bh,&bu,&bd,&bv);

    set_position(na,N-1,N,&bh,&bu,&bd,&bv);
    set_position(nb,N-2,N,&bh,&bu,&bd,&bv);

    set_position(N-1,N-1-ea,N,&bh,&bu,&bd,&bv);
    set_position(N-2,N-1-eb,N,&bh,&bu,&bd,&bv);

    set_position(N-1-sa,0,N,&bh,&bu,&bd,&bv);
    set_position(N-1-sb,1,N,&bh,&bu,&bd,&bv);

    if(sym == 0)
        assert(0);

    cuda_pre pre;
    pre.bh = bh;
    pre.bu = bu;
    pre.bd = bd;
    pre.bv = bv;
    pre.sym = sym;
    pre.result = 0; //TODO this should be not necessary
    
    return pre; 
}



int main(int argc, char ** argv){
    if(argc < 2){
        printf("Not enough arguments\n");
        return 1;
    }

    //int32_t N = atoi(argv[2]);
    FILE * fp = fopen(argv[1],"rb");
    if(fp == NULL){
        printf("\nCan not open file: %s\n",argv[1]);
        fflush(stdout);
        abort();
    }
    fseek(fp, 0L, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    if(sz % 4 != 0){
        printf("Wrong size.\n");
        abort();
    }
    int32_t preplacement_count = sz / 16;

    printf("\nPresets: Size: %ld Count: %d \n\n",sz,preplacement_count);

    preplacement * rawpre = (preplacement *) malloc(sz);
    int rr;
    if((rr = fread(rawpre,sizeof(preplacement),preplacement_count,fp)) != preplacement_count){
        printf("Read failed. Read: %d Should be:%d\n",rr,preplacement_count);
        abort();
    }
    fclose(fp);

    //start of cuda related stuff
    //CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1););
    
    
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,0);
    printf("\n \
            -----Geräteinfo-----\n \
            Name                : %s\n \
            Cudaversion         : %d.%d  \n \
            WarpSize            : %d\n \
            MultiProcessorCount : %d\n \
            MaxThreadsPerBlock  : %d\n \
            sharedMemPerBlock   : %ld\n \
            MaxThreadsDim       : (%d , %d , %d)\n \
            MaxGridSize         : (%d , %d , %d)\n ",
            device_prop.name,
            device_prop.major, device_prop.minor,
            device_prop.warpSize,
            device_prop.multiProcessorCount,
            device_prop.maxThreadsPerBlock,
            device_prop.sharedMemPerBlock,
            device_prop.maxThreadsDim[0],device_prop.maxThreadsDim[1],device_prop.maxThreadsDim[2],
            device_prop.maxGridSize[0],device_prop.maxGridSize[1],device_prop.maxGridSize[2]
            );

    cuda_pre * pre_host = (cuda_pre *) malloc(preplacement_count * sizeof(cuda_pre));
    cuda_pre * pre_device;
    
    for(uint32_t i=0;i<preplacement_count;i++){
        pre_host[i] = evaluate_preplacement(Ndef, be64toh(rawpre[i].m_spec));
    }
    free(rawpre);
    CUDA_CHECK(cudaMalloc((void**)&pre_device,preplacement_count * sizeof(cuda_pre)));
    
    CUDA_CHECK(cudaMemcpy(pre_device,pre_host,preplacement_count * sizeof(cuda_pre),cudaMemcpyHostToDevice));
    
    
    uint32_t dimGrid, dimBlock;
    dimBlock = Blocksizedef;
    dimGrid  = (int)(preplacement_count / dimBlock);
    dimGrid  = (preplacement_count % dimBlock > 0) ? dimGrid+1 : dimGrid; 
    dimGrid  = 13;

    if(dimGrid > device_prop.maxGridSize[0]){
        printf("\nGridsize %d is too big.\n",dimGrid);
        fflush(stdout);
        abort();
    }

    printf("\nKernel start.N=%d blocksize=%d gridsize=%d\n",
        Ndef,dimBlock,dimGrid);
    count_combinations<<<dimGrid,dimBlock>>>(pre_device,preplacement_count);
    CUDA_CHECK_KERNEL
    
    CUDA_CHECK(cudaMemcpy(pre_host,pre_device,preplacement_count * sizeof(cuda_pre),cudaMemcpyDeviceToHost));
    printf("\nKernel end.\n");

    CUDA_CHECK(cudaFree(pre_device));
    //end of cuda
    uint64_t found_combinations = 0;
    for(uint32_t i=0; i < preplacement_count; i++){
        found_combinations += pre_host[i].result * (1 << pre_host[i].sym);
    }
    printf("\nFound Combinations: %ld \n", found_combinations);
    free(pre_host);
}
