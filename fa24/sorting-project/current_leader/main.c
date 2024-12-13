#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>

#define ALLOC_NAME "default_allocator"

// You can change this value in Makefile.
#ifndef SIZE
Please define SIZE in Makefile.
#endif

typedef struct {
    int key;
    int value;
} ValueIndex;

int compare(const void *a, const void *b) {
    ValueIndex *vi1 = (ValueIndex *)a;
    ValueIndex *vi2 = (ValueIndex *)b;
    return (vi1->value - vi2->value); // Sort by value in ascending order
}

void sort(int *vi_array, int n) {
    // Sort the array using qsort
    return qsort(vi_array, n, sizeof(ValueIndex), compare);
}


int kernel_sort_bitonic(int argc, char **argv) {

  int rc;
  char *bin_path, *test_name;
  struct arguments_path args = {NULL, NULL};
  
  argp_parse(&argp_path, argc, argv, 0, 0, &args);
  bin_path = args.path;
  test_name = args.name;

  bsg_pr_test_info("Running kernel_sort_bitonic.\n");
  srand(time);
 
  // Initialize Device.
  hb_mc_device_t device;
  BSG_CUDA_CALL(hb_mc_device_init(&device, test_name, 0));

  hb_mc_pod_id_t pod;
  hb_mc_device_foreach_pod_id(&device, pod)
  {
    bsg_pr_info("Loading program for pod %d\n.", pod);
    BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
    BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

    ValueIndex * Unsorted_host = (ValueIndex*) malloc(sizeof(ValueIndex)*SIZE);
    ValueIndex * Sorted_host = (ValueIndex*) malloc(sizeof(ValueIndex)*SIZE);
    ValueIndex * Sorted_expected_host = (ValueIndex*) malloc(sizeof(ValueIndex)*SIZE);

    // Allocate a block of memory in host.
    for (int i = 0; i < SIZE; i++) {
      int rand_val = rand() % SIZE + 1;
      Unsorted_host[i].value = rand_val;
      Unsorted_host[i].key = i;
      Sorted_expected_host[i].value = rand_val;
      Sorted_expected_host[i].key = i;
    }
    
    sort(Sorted_expected_host, SIZE);

    printf("Unsorted values: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", Unsorted_host[i].value);
    }
    printf("\n");

    printf("Unsorted indices: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", Unsorted_host[i].key);
    }
    printf("\n");

    printf("Expected Sorted indices: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", Sorted_expected_host[i].key);
    }
    printf("\n");

    // Allocate a block of memory in device.
    eva_t Unsorted_device;
    BSG_CUDA_CALL(hb_mc_device_malloc(&device, SIZE * sizeof(ValueIndex), &Unsorted_device));
 
    // DMA Transfer to device.
    hb_mc_dma_htod_t htod_job [] = {
      {
        .d_addr = Unsorted_device,
        .h_addr = (void *) &Unsorted_host[0],
        .size = SIZE * sizeof(ValueIndex)
      }
    };

    BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, htod_job, 1));

    // CUDA arguments
    hb_mc_dimension_t tg_dim = { .x = bsg_tiles_X, .y = bsg_tiles_Y};
    hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};
    #define CUDA_ARGC 2
    uint32_t cuda_argv[CUDA_ARGC] = {Unsorted_device, SIZE};
    
    // Enqueue Kernel.
    BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_sort_bitonic", CUDA_ARGC, cuda_argv));
    
    // Launch kernel.
    //hb_mc_manycore_trace_enable((&device)->mc);
    BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));
    //hb_mc_manycore_trace_disable((&device)->mc);

    // Copy result and validate.
    hb_mc_dma_dtoh_t dtoh_job [] = {
      {
        .d_addr = Unsorted_device,
        .h_addr = (void *) &Sorted_host[0],
        .size = SIZE * sizeof(ValueIndex)
      }
    };

    BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, &dtoh_job, 1));

    for (int i = 0; i < SIZE; i++) {
      if (Sorted_expected_host[i].value != Sorted_host[i].value) {
        printf("FAIL Sorted_host[%d] = %x\n", i, Sorted_host[i].value);
        printf("FAIL Sorted_expected_host[%d] = %x\n", i, Sorted_expected_host[i].value);
        printf("FAIL Sorted_host = %x\n", Sorted_host);
        for (int i = 0; i < SIZE; i++) {
          printf("%d ", Sorted_host[i].key);
        }
        printf("\n");
        printf("FAIL Sorted_expected_host = %x\n", Sorted_expected_host);
        for (int i = 0; i < SIZE; i++) {
          printf("%d ", Sorted_expected_host[i].key);
        }
        printf("\n");

        printf("FAIL Sorted_host Values \n");
        for (int i = 0; i < SIZE; i++) {
          printf("%d ", Sorted_host[i].value);
        }
        printf("\n");
        printf("FAIL Sorted_expected_host Values \n");
        for (int i = 0; i < SIZE; i++) {
          printf("%d ", Sorted_expected_host[i].value);
        }
        printf("\n");
        BSG_CUDA_CALL(hb_mc_device_finish(&device));
        return HB_MC_FAIL;
      }
    }

    printf("PASSED GOLDEN TESTS");

    // Freeze tiles.
    BSG_CUDA_CALL(hb_mc_device_program_finish(&device));
  }

  BSG_CUDA_CALL(hb_mc_device_finish(&device));
  return HB_MC_SUCCESS; 
}


declare_program_main("sort_bitonic", kernel_sort_bitonic);
