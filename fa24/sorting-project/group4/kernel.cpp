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

// QuickSort function
__attribute__((noinline))
void quick_sort(KeyValuePair* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
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
    for (int i = 0; i < size; i++) {
        B[i] = A[i];
    }

    // Ensure all threads in the tile group have synchronized
    bsg_barrier_hw_tile_group_sync();

    // Perform QuickSort
    if (__bsg_id == 0) { // Only one thread performs the sorting
        quick_sort(B, 0, size - 1);
    }

    // Ensure sorted data is visible to all threads
    bsg_barrier_hw_tile_group_sync();

    // Debugging: End stats collection
    bsg_cuda_print_stat_kernel_end();

    return 0;
}