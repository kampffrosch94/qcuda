#include <stdio.h>
#include <stdlib.h>


//returns 1 if current field state is valid else returns 0
int checkfield(char * rows, int fieldlength){
    //same row is not possible because of the way the field
    //is saved => doesn't need to be checked
    for(int i = 0; i<fieldlength-1;i++){
        int e = i + 1;
        while(e < fieldlength){
            if(rows[e] == -1){ //row not yet set
                break; 
            }
            if(rows[i] == rows[e]){ //same column
                return 0;
            }
            if(abs(rows[i] - rows[e]) == e-i){ //same diagonal
                return 0;
            }
            e++;
        }
    }
    return 1;
}

int main(int argc, char ** argv){
    int fieldlength = 4; //width or hight of the chess field
    char * rows = (char *) calloc(fieldlength,sizeof(char));
    int currentrow = 0;

    for (int i = 0; i<fieldlength;i++){
        rows[i] = -1;
    }
    
    while(currentrow < fieldlength){
        rows[currentrow]++;
        if(rows[currentrow] >= fieldlength){//backtracking
            rows[currentrow] = -1;
            currentrow--;
            while((currentrow >= 0) && //multistepbacktracking
                    (rows[currentrow] >= fieldlength - 1)){
                rows[currentrow] = -1;
                currentrow--;
                printf("Multi");
            }

            printf("Track back\n");
            if(currentrow < 0){
                printf("\nNo solution found.\n");
                return 0;
            }
        }
        else if(checkfield(rows,fieldlength)){
            currentrow++;
        }   
    }

    printf("Result:\n\n");
    for(int i = 0; i<fieldlength;i++){
        for(int e = 0; e<fieldlength;e++){
            if(e == rows[i])
                printf("X");
            else
                printf("O");
        }
        printf("\n");
    }
}
