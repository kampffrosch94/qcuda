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

uint64_t count_combinations(int32_t N){//3 < N < 64
    uint64_t * cols = calloc(N,sizeof(int64_t));
    //the field on which the queens are set, split in columns
    //onehot coded for the row in which the queen is placed
    //0 means unset
    //int64_t bv = 0; //block vertical has to be preset TODO
    uint64_t bh = 0; //block horizontal
    //count from bottom to top
    uint64_t bu = 0; //element 0 on the lower left
    uint64_t bd = 0; //element 0 on the lower right

    //blockvectors for a new position
    uint64_t tbh; 
    uint64_t tbu; 
    uint64_t tbd; 

    int64_t * free_vectors = malloc(N*sizeof(int64_t)); 
    //has 1 in positions which are free in the
    //current col (opposite to bh)

    uint64_t relevant_bits = (1 << N) - 1;

    uint32_t cc = 0; //current column

    uint64_t found_combinations = 0;

    free_vectors[0] = ~(bh | (bu >> cc) | (bd >> (N -1 - cc))) & relevant_bits;

    while(1){
        if(cols[cc] > 0){
            //delete the old placement in the blockvectors (if any)
            tbh = cols[cc];
            tbu = tbh << cc;
            tbd = tbh << (N - 1 - cc);

            bh -= tbh;
            bu -= tbu;
            bd -= tbd;
        }


        if(free_vectors[cc] == 0){ //track back
            if(cc == 0){ //stop
                break;
            }
            cols[cc] = 0;
            cc--;
        } else {
            //find the next new placement in the current col
            cols[cc] = free_vectors[cc] & -free_vectors[cc];
            //remove it from the free_vector
            free_vectors[cc] -= cols[cc];
            //set the new placement in the blockvectors
            tbh = cols[cc];
            tbu = tbh << cc;
            tbd = tbh << (N - 1 - cc);

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
                free_vectors[cc] = ~(bh | (bu >> cc) | (bd >> (N -1 - cc))) 
                    & relevant_bits;
            }
        }
    }
    free(cols);
    free(free_vectors);
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
