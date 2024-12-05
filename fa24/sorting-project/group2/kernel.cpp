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
} ValueKeyPair;
#define BITS_PER_PASS 4  // Sort by 4 bits at a time
#define NUM_BUCKETS (1 << BITS_PER_PASS)  // 16 buckets for 4 bits
#define MAX_BITS 32  // Sorting 32-bit integers
// Local histogram
__attribute__((section(".dram"))) int local_count[NUM_BUCKETS * bsg_tiles_X * bsg_tiles_Y];
__attribute__((section(".dram"))) int prefix_sum[NUM_BUCKETS];
__attribute__((section(".dram"))) int prefix_sum_buffer[NUM_BUCKETS];
void blellochScan(int *d_array, int size) {
  int block_size = bsg_tiles_X * bsg_tiles_Y;
  int tid = __bsg_id;
  if (__bsg_id == 0) {
    for (int i = 0; i < size; i++) {
      prefix_sum_buffer[i] = d_array[i];
    }
  }
  bsg_barrier_hw_tile_group_sync();
  // Upsweep
  for (int step = 1; step < size; step *= 2) {
    int index = (tid + 1) * step * 2 - 1;
    if (index < size) {
      prefix_sum_buffer[index] += prefix_sum_buffer[index - step];
    }
    bsg_barrier_hw_tile_group_sync();
  }
  // Clear the last element to prepare for downsweep
  if (tid == (block_size) - 1) {
      prefix_sum_buffer[size - 1] = 0;
  }
  bsg_barrier_hw_tile_group_sync();
  // Downsweep phase
  for (int step = size / 2; step > 0; step /= 2) {
      int index = (tid + 1) * step * 2 - 1;
      if (index < size) {
          int temp = prefix_sum_buffer[index];
          prefix_sum_buffer[index] += prefix_sum_buffer[index - step];
          prefix_sum_buffer[index - step] = temp;
      }
      bsg_barrier_hw_tile_group_sync();
  }
  // Write back to global memory
  if (__bsg_id == 0) {
    for (int i = 0; i < size; i += 1) {
      d_array[i] = prefix_sum_buffer[i];
    }
  }
}
void warmup(ValueKeyPair *d_array, int size) {
    int tid = __bsg_id;
    int block_size = bsg_tiles_X * bsg_tiles_Y;
    // Warm up the cache by accessing the elements
    for (int i = tid; i < size; i += block_size) {
        uint32_t value = d_array[i].value;
        asm volatile ("" : : "r" (value) : "memory");
    }
    bsg_barrier_hw_tile_group_sync();
}
void radixSort(ValueKeyPair *d_array, ValueKeyPair *d_buffer, int size, int bit_offset) {
    int tid = __bsg_id;
    int block_size = bsg_tiles_X * bsg_tiles_Y;
    // Initialize local count
    for (int i = tid; i < NUM_BUCKETS * block_size; i += block_size) {
        local_count[i] = 0;
    }
    bsg_barrier_hw_tile_group_sync();
    // Build histogram for this digit
    for (int i = __bsg_id; i < size; i += block_size) {
        uint32_t value = d_array[i].value;
        int digit = __bsg_id + block_size * ((value >> bit_offset) & (NUM_BUCKETS - 1));
        bsg_amoadd(&local_count[digit], 1);
    }
    bsg_barrier_hw_tile_group_sync();
    // Perform an exclusive scan (prefix sum) on the histogram
    for (int i = tid; i < NUM_BUCKETS; i += block_size) {
      prefix_sum[i] = 0;
    }
    bsg_barrier_hw_tile_group_sync();
    // Sum up for different tiles
    for (int bucket = 0; bucket < NUM_BUCKETS; bucket++) {
      int added_occurence = local_count[tid + bucket * block_size];
      bsg_amoadd(&prefix_sum[bucket], added_occurence);
    }
    bsg_barrier_hw_tile_group_sync();
    // Scan for prefix sum
    blellochScan(prefix_sum, NUM_BUCKETS);
    // Rearrange elements based on the prefix sum
    if (__bsg_id == 0) {
      for (int i = 0; i < size; i += 1) {
        uint32_t value = d_array[i].value;
        int digit = (value >> bit_offset) & (NUM_BUCKETS - 1);
        d_buffer[prefix_sum[digit]] = d_array[i];
        bsg_amoadd(&prefix_sum[digit], 1);
      }
    }
    bsg_barrier_hw_tile_group_sync();
    // Copy sorted data back to the original array
    if (__bsg_id == 0) {
      for (int i = 0; i < size; i += 1) {
        d_array[i] = d_buffer[i];
      }
    }
}
extern "C" __attribute__ ((noinline))
int kernel_sort_radix(ValueKeyPair * Unsorted, ValueKeyPair * Sorted, int N) {
  bsg_barrier_hw_tile_group_init();
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();
  // Warm up the cache
  warmup(Unsorted, N);
  bsg_barrier_hw_tile_group_sync();
  for (int bit = 0; bit < 32; bit += BITS_PER_PASS) {
      radixSort(Unsorted, Sorted, N, bit);
      bsg_barrier_hw_tile_group_sync();
  }
  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();
  return 0;
}