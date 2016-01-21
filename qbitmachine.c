#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define VERBOSE 0

typedef struct {
    uint64_t m_spec;
    uint64_t m_sol;
} preplacement;

void print_field(uint64_t * cols, int32_t N){
    for(int row = 0; row < N; row++){
        for(int col = 0; col < N; col++){
            if(cols[col] == 1 << (N -1 -row)){
                printf("X");
            } else {
                printf("O");
            }
        }
        printf("\n");
    }
    printf("\n");
}

uint64_t count_combinations(int32_t N,int32_t L,uint64_t bh,uint64_t bu,
        uint64_t bd,uint64_t bv){//3 < N < 64

    uint64_t * cols = calloc(N,sizeof(int64_t));
    //the field on which the queens are set, split in columns
    //onehot coded for the row in which the queen is placed
    //0 means unset

    //int64_t bv = 0; //block vertical has to be preset
    //uint64_t bh = 0; //block horizontal
    //count from bottom to top
    //uint64_t bu = 0; //element 0 on the lower left
    //uint64_t bd = 0; //element 0 on the lower right

    //blockvectors for a new position
    uint64_t tbh; 
    uint64_t tbu; 
    uint64_t tbd; 

    int64_t * free_vectors = calloc(N,sizeof(int64_t)); 
    //has 1 in positions which are free in the
    //current col (opposite to bh)

    uint64_t relevant_bits = (1 << N) - 1;

    uint32_t cc = L; //current column

    uint64_t found_combinations = 0;

    free_vectors[L] = ~(bh | (bu >> cc) | (bd >> (N -1 - cc))) & relevant_bits;

    while(1){
        while(((bv >> cc)&1)&&(cc <= N - 1 - L)){
            cc++;
        }
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
            if(cc == L){ //stop
                break;
            }
            cols[cc] = 0;
            cc--;
            while(((bv >> cc)&1)&&(cc>=L)){
                cc--;
            }
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

            if(cc == N - 1 - L){ 
                found_combinations++;
#if (VERBOSE > 0)
                print_field(cols,N);
                //printf("Free_vector: %s\n",byte_to_binary(free_vector));
                //printf("bd: %s\n",byte_to_binary(bd));
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

void set_position(int32_t x,int32_t y, int32_t N,uint64_t * bh, 
        uint64_t * bu, uint64_t * bd, uint64_t * bv){
    uint64_t tbh = 1 << y;
    uint64_t tbu = tbh << x;
    uint64_t tbd = tbh << (N - 1 - x);
    uint64_t tbv = 1 << x;

    *bh += tbh;
    *bu += tbu;
    *bd += tbd;
    *bv += tbv;
}

uint64_t evaluate_preplacement(int32_t N, preplacement pre){
    uint8_t wa,wb,na,nb,ea,eb,sa,sb,sym;
    uint64_t spec = pre.m_spec;

    wa = spec >> 60;
    wb = spec >> 55;
    na = spec >> 50;
    nb = spec >> 45;
    ea = spec >> 40;
    eb = spec >> 35;
    sa = spec >> 30;
    sb = spec >> 25;
    sym = spec >> 23;

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

    if(sym == 3)//None
        return count_combinations(N,2,bh,bu,bd,bv);
    else if(sym == 2)//Point
        return count_combinations(N,2,bh,bu,bd,bv)*2;
    else if(sym == 1)//Symmetrie
        return count_combinations(N,2,bh,bu,bd,bv)*4;
    else 
        return count_combinations(N,2,bh,bu,bd,bv);
        //assert(0);
}



int main(int argc, char ** argv){
    if(argc < 3){
        printf("Not enough arguments\n");
        return 1;
    }

    FILE * fp = fopen(argv[1],"rb");
    int32_t N = atoi(argv[2]);
    fseek(fp, 0L, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    if(sz % 4 != 0){
        printf("Wrong size.\n");
        abort();
    }
    int32_t count = sz / 16;

    printf("\nSize: %ld Count: %d \n",sz,count);

    preplacement * pre = malloc(sz);
    int rr;
    if((rr = fread(pre,sizeof(preplacement),count,fp)) != count){
        printf("Read failed. Read: %d Should be:%d\n",rr,count);
        abort();
    }

    uint64_t found_combinations = 0;
    for(int i=0;i<count;i++){
        found_combinations += evaluate_preplacement(N, pre[i]);
    }
    //printf("\nFound Combinations: %" PRIu64 "\n", count_combinations(N));
}
