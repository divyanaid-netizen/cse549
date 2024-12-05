#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <math.h>

typedef struct {
    int key;
    int value;
} ValueIndex;

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

void mergeSort(ValueIndex *arr, ValueIndex *temp, int block_offset, int step, int numElements) {
  int left = block_offset + __bsg_id * step * 2;  // Start of left subarray
  int right = left + step;                        // Start of right subarray
  int end = min(left + step * 2, numElements);    // End of subarray region

  if (right >= numElements) return; // Nothing to merge if right subarray is out of bounds

  int i = left;    // Walking Pointer for left subarray
  int j = right;   // Walking Pointer for right subarray
  int k = left;    // Walking Pointer for temporary merge array

  // Merge the two halves
  while (i < right && j < end) {
      if (arr[i].value <= arr[j].value) {
          temp[k++] = arr[i++];
      } else {
          temp[k++] = arr[j++];
      }
  }
  // Copy final elements from left half
  while (i < right) {
      temp[k++] = arr[i++];
  }

  // Copy final elements from right half
  while (j < end) {
      temp[k++] = arr[j++];
  }

  // Copy data back to the original array
  for (int m = left; m < end; m++) {
      arr[m] = temp[m];
  }
}

extern "C" __attribute__ ((noinline))
int kernel_sort(ValueIndex * Unsorted, ValueIndex * Sorted, int N) {

  bsg_barrier_hw_tile_group_init();
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  for (int step = 1; step < N; step *= 2) {
    int block_stride = (bsg_tiles_X * bsg_tiles_Y * step * 2);
    int max_blocks = N / block_stride + ( N % block_stride != 0 );
    for (int block_step = 0; block_step < max_blocks; block_step++) {
      int block_offset = block_step * bsg_tiles_X * bsg_tiles_Y * step * 2;
      mergeSort(Unsorted, Sorted, block_offset, step, N);
      bsg_barrier_hw_tile_group_sync();
    }
  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();

  return 0;
}
