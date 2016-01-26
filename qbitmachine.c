#define DEBUG 0
#define WITHOUTPRESET 0
#define _BSD_SOURCE 
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <endian.h>


typedef struct {
    uint64_t m_spec;
    uint64_t m_sol;
} preplacement;

uint64_t count_combinations(int32_t N,int32_t L,uint64_t bh,uint64_t bu,
        uint64_t bd,uint64_t bv){//3 < N < 64

    //uint64_t bv = 0; //block vertical has to be preset
    //uint64_t bh = 0; //block horizontal; count from bottom to top
    //uint64_t bu = 0; //element 0 on the lower left
    //uint64_t bd = 0; //element 0 on the lower right

    uint64_t * cols;
    //the field on which the queens are set, split in columns
    //onehot coded for the row in which the queen is placed
    //0 means unset
    int64_t * free_vectors;
    //has 1 in positions which are free in the
    //current col (opposite to bh)

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

#ifdef DEBUG
#if DEBUG > 0
    if(start != L)
        printf("start changed.\n");
    if(end != N-1-L)
        printf("end changed.\n");
#endif
    if(start < L)
        printf("start is too small.\n");
    if(end >= N)
        printf("end is too big.\n");
    if((start > end) || (end < start))
        printf("WTH!\n");
    fflush(stdout);
#endif

    cols = calloc(N,sizeof(int64_t));
    free_vectors = calloc(N,sizeof(int64_t)); 
    
    int32_t cc = start; //current column
    
    free_vectors[cc] = ~(bh | (bu >> cc) | (bd >> (N -1 - cc))) 
        & relevant_bits;

    while(1){
#ifdef DEBUG
        if((cc<start)||(cc > end))
            printf("cc out of bounds\n");
#endif

        if(cols[cc] > 0){
            //delete the old placement in the blockvectors (if any)
            bh ^= cols[cc];
            bu ^= cols[cc] << cc;
            bd ^= cols[cc] << (N - 1 - cc);
        }

        if(free_vectors[cc] == 0){ //track back
            if(cc == start){ //stop
                break;
            }
            cols[cc] = 0;
            cc--;
            while(((bv >> cc)&1)){
                cc--;
            }
        } else {
            //find the next new placement in the current col
            cols[cc] = free_vectors[cc] & -free_vectors[cc];
            //remove it from the free_vector
            free_vectors[cc] ^= cols[cc];
            //set the new placement in the blockvectors
            bh ^= cols[cc];
            bu ^= cols[cc] << cc;
            bd ^= cols[cc] << (N - 1 - cc);

            if(cc == end){ 
                found_combinations++;
            } else {
                cc++;
                while(((bv >> cc)&1)){
                    cc++;
                }
                free_vectors[cc] = ~(bh | (bu >> cc) | (bd >> (N - 1 - cc))) 
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

    *bh |= tbh;
    *bu |= tbu;
    *bd |= tbd;
    *bv |= tbv;

#if (DEBUG > 1)
    printf("(%d,%d)\n",x,y);
#endif
}

uint64_t evaluate_preplacement(int32_t N, uint64_t spec){
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

    int64_t result = count_combinations(N,2,bh,bu,bd,bv);

#if (DEBUG > 0)
    uint64_t relevant_bits = (1 << N) - 1;
    int8_t cc = 2;
    uint64_t free_vector = (~(bh | (bu >> cc) | (bd >> (N - 1 - cc)))) 
        & relevant_bits;
    uint64_t first_free = free_vector & -free_vector;
    if(0 != result){
        printf("First Free: %ld \n",first_free);
        printf("Sym: %d Result: %ld\n------------------\n\n",sym,result);
    }
#endif

    return result*(1 << sym); 
}



int main(int argc, char ** argv){
    if(argc < 3){
        printf("Not enough arguments\n");
        return 1;
    }

    int32_t N = atoi(argv[2]);
#if WITHOUTPRESET > 0
    int64_t result = count_combinations(N,0,0,0,0,0);
    printf("\nN=%d Without presets: %ld\n",N,result);
#endif

#if WITHOUTPRESET == 0
    FILE * fp = fopen(argv[1],"rb");
    fseek(fp, 0L, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    if(sz % 4 != 0){
        printf("Wrong size.\n");
        abort();
    }
    int32_t count = sz / 16;

    printf("\nPresets: Size: %ld Count: %d \n\n",sz,count);

    preplacement * pre = malloc(sz);
    int rr;
    if((rr = fread(pre,sizeof(preplacement),count,fp)) != count){
        printf("Read failed. Read: %d Should be:%d\n",rr,count);
        abort();
    }

    uint64_t found_combinations = 0;
    for(int i=0;i<count;i++){
#if DEBUG > 1
        printf("\n%d\n",i+1);
#endif
        found_combinations += evaluate_preplacement(N, be64toh(pre[i].m_spec));
    }
    printf("\nFound Combinations: %" PRIu64 "\n", found_combinations);
#endif
}
