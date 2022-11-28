#define FULL_WARP_MASK 0xffffffff

#define NNZ_PER_WG 256 ///< Should be power of two

#define VECTOR_VS_VECTORL_NNZ_THRESHOLD 64

#define WARP_SIZE MIN(32, NNZ_PER_WG)

#define BOUND_REDUCTION_NUM_THREADS 256

#define MAX_NUM_ROUNDS 100

#define SHARED_MEM_THREADS 8

#define BOUND_COPY_NUM_THREADS 256

#define GDP_INF 1e20

#define GDP_EPS 1e-6

