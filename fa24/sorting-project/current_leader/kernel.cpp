#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <math.h>
#include "bsg_manycore_atomic.h"
#include "bsg_cuda_lite_barrier.h"
#include <atomic>
#include <algorithm>

typedef struct {
    uint32_t key;
    uint32_t value;
} ValueIndex;

void bitonicSortStep(ValueIndex *data, int n, int j, int k) {
    uint32_t elements_per_tile = n / (bsg_tiles_X * bsg_tiles_Y);

    for (uint32_t e = 0; e < elements_per_tile; ++e) {
        uint32_t i = __bsg_id * elements_per_tile + e;
        if (i >= n) return;
        uint32_t ixj = i ^ j;

        if (ixj > i) {
          if ((i & k) == 0) { // Sort ascending
              if (data[i].value > data[ixj].value) {
                  ValueIndex temp = data[i];
                  data[i] = data[ixj];
                  data[ixj] = temp;
              }
          } else { // Sort descending
              if (data[i].value < data[ixj].value) {
                  ValueIndex temp = data[i];
                  data[i] = data[ixj];
                  data[ixj] = temp;
              }
          }
        }
    }
}

void warmup(ValueIndex *d_array, int size) {
    int tid = __bsg_id;
    int block_size = bsg_tiles_X * bsg_tiles_Y;
    // Warm up the cache by accessing the elements
    for (int i = tid; i < size; i += block_size) {
        uint32_t value = d_array[i].value;
        asm volatile ("" : : "r" (value) : "memory");
    }
}

extern "C" __attribute__ ((noinline))
int kernel_sort_bitonic(ValueIndex *Unsorted, int N) {
  bsg_barrier_hw_tile_group_init();
  // Cache warming before sorting
  warmup(Unsorted, N);
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();
  
  for (int k = 2; k <= N; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
          bitonicSortStep(Unsorted, N, j, k);
          bsg_barrier_hw_tile_group_sync();
      }
  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();
  return 0;
}