#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>

typedef struct {
    int id;
    int data;
} KeyValuePair;

// Swapping function
__attribute__((noinline))
void swap(KeyValuePair* a, KeyValuePair* b) {
    KeyValuePair temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function
__attribute__((noinline))
int partition(KeyValuePair* arr, int low, int high) {
    KeyValuePair pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j].data <= pivot.data) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

// Serial QuickSort fallback
__attribute__((noinline))
void quick_sort(KeyValuePair* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// Parallel QuickSort function
__attribute__((noinline))
void parallel_quick_sort(KeyValuePair* arr, int low, int high, int depth) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (depth > 0) {
            // Parallelizing work based on blocks
            int block_size = (high - low + 1) / (__bsg_grid_dim_x * __bsg_grid_dim_y);

            if (block_size > 1) {
                // Left thread handling the left partition
                if (__bsg_id * block_size < high) {
                    parallel_quick_sort(arr, low, pi - 1, depth - 1);
                }

                // Right thread handling the right partition
                if (__bsg_id * block_size < high) {
                    parallel_quick_sort(arr, pi + 1, high, depth - 1);
                }
            }
        } else {
            // Use serial QuickSort when depth is zero
            quick_sort(arr, low, pi - 1);
            quick_sort(arr, pi + 1, high);
        }
    }
}

// Kernel function
extern "C" __attribute__((noinline))
int kernel_sort(KeyValuePair* A, KeyValuePair* B, int size) {
    // Initialize barrier
    bsg_barrier_hw_tile_group_init();
    bsg_barrier_hw_tile_group_sync();

    // Debugging: Start stats collection
    bsg_cuda_print_stat_kernel_start();

    // Copy input array to output array
    for (int i = __bsg_id; i < size; i += __bsg_grid_dim_x * __bsg_grid_dim_y) {
        B[i] = A[i];
    }

    // Ensure all threads in the tile group have synchronized
    bsg_barrier_hw_tile_group_sync();

    // Perform Parallel QuickSort
    if (__bsg_id == 0) { // Only the master thread initiates parallel sorting
        parallel_quick_sort(B, 0, size - 1, 3); // Depth controls parallelization level
    }

    // Ensure sorted data is visible to all threads
    bsg_barrier_hw_tile_group_sync();

    // Debugging: End stats collection
    bsg_cuda_print_stat_kernel_end();

    return 0;
}
