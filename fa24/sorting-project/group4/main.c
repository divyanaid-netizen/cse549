#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <bsg_manycore_regression.h>

#define ALLOCATOR_NAME "default_allocator"
#undef ARRAY_SIZE               // Remove any previous definition of ARRAY_SIZE
#define ARRAY_SIZE (1 << 14)    // Define array size as 2^20

typedef struct {
    int id;
    int data;
} KeyValuePair;

int compare_keys(const void *a, const void *b) {
    KeyValuePair *pair1 = (KeyValuePair *)a;
    KeyValuePair *pair2 = (KeyValuePair *)b;
    return (pair1->data - pair2->data);
}

void sort_key_value_pairs(KeyValuePair *kv_array, int count) {
    qsort(kv_array, count, sizeof(KeyValuePair), compare_keys);
}

int execute_sort_kernel(int argc, char **argv) {
    int status_code;
    char *binary_path, *test_label;
    struct arguments_path arguments = {NULL, NULL};
    argp_parse(&argp_path, argc, argv, 0, 0, &arguments);
    binary_path = arguments.path;
    test_label = arguments.name;
    bsg_pr_test_info("Executing sort kernel.\n");
    srand(time(NULL));

    // Initialize the device
    hb_mc_device_t accelerator;
    BSG_CUDA_CALL(hb_mc_device_init(&accelerator, test_label, 0));

    hb_mc_pod_id_t pod_id;
    hb_mc_device_foreach_pod_id(&accelerator, pod_id) {
        bsg_pr_info("Loading program for pod %d\n", pod_id);
        BSG_CUDA_CALL(hb_mc_device_set_default_pod(&accelerator, pod_id));
        BSG_CUDA_CALL(hb_mc_device_program_init(&accelerator, binary_path, ALLOCATOR_NAME, 0));

        KeyValuePair *input_array = (KeyValuePair *)malloc(sizeof(KeyValuePair) * ARRAY_SIZE);
        KeyValuePair *sorted_output = (KeyValuePair *)malloc(sizeof(KeyValuePair) * ARRAY_SIZE);
        KeyValuePair *expected_output = (KeyValuePair *)malloc(sizeof(KeyValuePair) * ARRAY_SIZE);

        // Initialize random data
        for (int i = 0; i < ARRAY_SIZE; i++) {
            int random_value = rand() % ARRAY_SIZE + 1;
            input_array[i].data = random_value;
            input_array[i].id = i;
            expected_output[i].data = random_value;
            expected_output[i].id = i;
        }

        // Software sorting (reference)
        sort_key_value_pairs(expected_output, ARRAY_SIZE);

        // Print input data
        printf("Input Array: \n");
        for (int i = 0; i < ARRAY_SIZE && i < 10; i++) { // Limit printing to first 10 elements
            printf("%d %d\n", input_array[i].data, input_array[i].id);
        }
        printf("\n");

        printf("Expected sorted array: \n");
        for (int i = 0; i < ARRAY_SIZE && i < 10; i++) { // Limit printing to first 10 elements
            printf("%d %d\n", expected_output[i].data, expected_output[i].id);
        }
        printf("\n");

        // Allocate memory on the device
        eva_t input_device, output_device;
        BSG_CUDA_CALL(hb_mc_device_malloc(&accelerator, ARRAY_SIZE * sizeof(KeyValuePair), &input_device));
        BSG_CUDA_CALL(hb_mc_device_malloc(&accelerator, ARRAY_SIZE * sizeof(KeyValuePair), &output_device));

        // DMA Transfer to device
        hb_mc_dma_htod_t dma_to_device[] = {
            {
                .d_addr = input_device,
                .h_addr = (void *)&input_array[0],
                .size = ARRAY_SIZE * sizeof(KeyValuePair)
            }
        };
        BSG_CUDA_CALL(hb_mc_device_dma_to_device(&accelerator, dma_to_device, 1));

        // Launch kernel
        hb_mc_dimension_t tile_dim = {.x = bsg_tiles_X, .y = bsg_tiles_Y};
        hb_mc_dimension_t grid_dim = {.x = 1, .y = 1};
        #define KERNEL_ARG_COUNT 3
        uint32_t kernel_args[KERNEL_ARG_COUNT] = {input_device, output_device, ARRAY_SIZE};

        BSG_CUDA_CALL(hb_mc_kernel_enqueue(&accelerator, grid_dim, tile_dim, "kernel_sort", KERNEL_ARG_COUNT, kernel_args));
        hb_mc_manycore_trace_enable((&accelerator)->mc);
        BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&accelerator));
        hb_mc_manycore_trace_disable((&accelerator)->mc);

        // DMA Transfer from device
        hb_mc_dma_dtoh_t dma_to_host[] = {
            {
                .d_addr = output_device,
                .h_addr = (void *)&sorted_output[0],
                .size = ARRAY_SIZE * sizeof(KeyValuePair)
            }
        };
        BSG_CUDA_CALL(hb_mc_device_dma_to_host(&accelerator, dma_to_host, 1));

        printf("Sorted Array: \n");
        for (int i = 0; i < ARRAY_SIZE && i < 10; i++) { // Limit printing to first 10 elements
            printf("%d %d\n", sorted_output[i].data, sorted_output[i].id);
        }
        printf("\n");

        // Validate results
        int is_valid = 1;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            if (expected_output[i].data != sorted_output[i].data) {
                printf("ERROR: Mismatch at index %d: expected %d, got %d\n",
                       i, expected_output[i].data, sorted_output[i].data);
                is_valid = 0;
            }
        }

        if (is_valid) {
            printf("Validation PASSED: Hardware sorting matches software sorting.\n");
        } else {
            printf("Validation FAILED: Hardware sorting does not match software sorting.\n");
        }

        free(input_array);
        free(sorted_output);
        free(expected_output);

        BSG_CUDA_CALL(hb_mc_device_program_finish(&accelerator));
    }

    BSG_CUDA_CALL(hb_mc_device_finish(&accelerator));
    return HB_MC_SUCCESS;
}

declare_program_main("sort", execute_sort_kernel);
