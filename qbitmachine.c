#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>

#define VERBOSE 0

const char *byte_to_binary(int x)
{
    static char b[8];
    b[0] = '\0';

    int z;
    for (z = 64; z > 0; z >>= 1)
    {
        strcat(b, ((x & z) == z) ? "1" : "0");
    }
    return b;
}

void print_field(int8_t * cols, int32_t N){
    for(int row = 0; row < N; row++){
        for(int col = 0; col < N; col++){
            if(N - cols[col] - 1 == row){
                printf("X");
            } else {
                printf("O");
            }
        }
        printf("\n");
    }
    for(int i = 0; i < N; i++){
        printf("%"PRId8" , ",cols[i]);
    }
    //printf("\n--------------\n");
    printf("\n");
}

uint64_t count_combinations(int32_t N){
    int8_t * cols = calloc(N,sizeof(int8_t));
        //the field on which the queens are set, split in columns
    //int64_t bv = 0; //block vertical has to be preset
    int64_t bh = 0; //block horizontal
                 //count from bottom to top
    int64_t bu = 0; //element 0 on the lower left
    int64_t bd = 0; //element 0 on the upper left

    //blockvectors for a new position
    int64_t tbh; 
    int64_t tbu; 
    int64_t tbd; 

    int64_t free_vector; //has 1 in positions which are free in the
                         //current col (opposite bh)

    int32_t cc = 0; //current column

    uint64_t found_combinations = 0;
    
    for(int32_t i=0;i<N;i++){
        cols[i] = -1; //this value means currently not set
    }

    while(1){
        if(cols[cc] > -1){
            //delete the old placement in the blockvectors (if any)
            tbh = 1 << cols[cc];
            tbu = 1 << (cols[cc] + cc);
            tbd = 1 << (cols[cc] + N - 1 - cc);
            
            bh -= tbh;
            bu -= tbu;
            bd -= tbd;
        }

        cols[cc]++;

        //find the next new placement in the current col
        free_vector = ~(bh | (bu >> cc) | (bd >> (N -1 - cc)));
        while( (cols[cc] < N) && 
                !((free_vector >> cols[cc]) & 1) ){
            cols[cc]++;
        } //postcondition: cols[cc] is at free position or equal to N

        if(cols[cc] >= N){ //track back
            if(cc == 0){ //stop
                break;
            }
            cols[cc] = -1;
            cc--;
        } else {
            //set the new placement in the blockvectors
            tbh = 1 << cols[cc];
            tbu = 1 << (cols[cc] + cc);
            tbd = 1 << (cols[cc] + N - 1 - cc);

            bh += tbh;
            bu += tbu;
            bd += tbd;
            
            if(cc == N - 1){
                found_combinations++;
#if (VERBOSE > 0)
                print_field(cols,N);
                //printf("Free_vector: %s\n",byte_to_binary(free_vector));
                printf("bd: %s\n",byte_to_binary(bd));
#endif
            } else {
                cc++;
            }
        }
    }
    free(cols);
    return found_combinations;
}

int main(int argc, char ** argv){
    if(argc < 2){
        printf("Not enough arguments\n");
        return 1;
    }
    int32_t N = atoi(argv[1]);
    printf("\nFound Combinations: %" PRIu64 "\n", count_combinations(N));
}
