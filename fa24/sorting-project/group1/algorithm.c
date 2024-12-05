/*Contains a sorting algorithm to verify correctness from HB kernel*/

#include <math.h>
#include <stdlib.h>

/*Simple Insertion sort algorithm*/
float* insertionSort(float *A, int n){
    /*Initialize sorted array*/
    float* B = (float*) malloc(n*sizeof(float));
    memcpy(B, A, n);

    int i, j;
    for (i = 1; i < n; i++){
        float key = B[i];

        /*Move key to correct position by shifting all previous entries that are greater than key one index ahead*/
        for(j = i-1; j >= 0 && B[j] > key; j--){
            B[j+1] = B[j];
        }
        B[j+1] = key;
    }

    return B;
}