#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>

/*Sorting loop with Unroll Factor of 0*/

#ifdef WARM_CACHE
__attribute__((noinline))
static void warmup(float *A, int N)
{
  for (int i = __bsg_id*CACHE_LINE_WORDS; i < N; i += bsg_tiles_X*bsg_tiles_Y*CACHE_LINE_WORDS) {
      asm volatile ("lw x0, %[p]" :: [p] "m" (A[i]));
  }
  bsg_fence();
}
#endif


/*Sort vector A of length N*/
extern "C" __attribute__ ((noinline))
int
kernel_sorter(float * A, int N) {

  bsg_barrier_hw_tile_group_init();
#ifdef WARM_CACHE
  warmup(A, N);
#endif
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  /*Each tile sorts a subsection of the vector*/
  int len = N / (bsg_tiles_X*bsg_tiles_Y);
  float *myA = &A[__bsg_id*len];

  /*Unroll the loop by a factor of 0*/
  for (int i = 0; i < len; i+=1) {
    /*Load all operands*/
    register float A00 = myA[i+0];
    bsg_fence();  /*Make loads non-blocking*/
    register float A01 = myA[i+1];
    register float temp = A00;

    asm volatile("": : :"memory");
    /*Simple if-else statement for comparison*/
    if(A00 > A01){
      A00 = A01;
      A01 = temp;
    }

    /*Store all results*/
    myA[i+0] = A00;
    bsg_fence();  /*Make stores non-blocking*/
    myA[i+1] = A01;
  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();

  return 0;
}
