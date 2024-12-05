
Rajeev B. Botadra
CSE549 Autumn 2024
Professor Taylor

# Final Project: Parallelized Sorting Kernel

## Algorithm 

The current kernel is a naive bubble sort algorithm that partitions the array into 256 sectors and assigns each sector to a compute tile. The sorted sectors are then joined into arrays double the size and resorted (only utiling half the tiles as the sort completes over the entire array). The loop is not unrolled and the algorithm is very poorly optimized.

## Cycle Counts & Geomean
| Size      | Cycle Count   |
| --------- | ------------- |
| 2^14      | 223,604       |
| 2^16      | 1,201,118     |
| 2^18      | 20,543,012   |
| 2^20      | DNF           |
| 2^22      | DNF           |
| 2^24      | DNF           |

Geomean = 1,767,026.738 Cycles