
#define VALUE_TYPE float
#define BLOCK_SIZE 16
__kernel void cdot(const __global VALUE_TYPE* A,
                    int hA, int wA,
                      const __global VALUE_TYPE* B,
                      int hB, int wB,
                      __global VALUE_TYPE* C) {
  int gY = get_group_id(0);
  int gX = get_group_id(1);
  int lY = get_local_id(0);
  int lX = get_local_id(1);

  float Csub = 0;



    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = wA * BLOCK_SIZE * gY;


    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + wA - 1;


    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;



    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = BLOCK_SIZE * gX;    


    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;



    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
    {

        // Declaration of the local memory array As 
        // used to store the sub-matrix of A
        __local float As[BLOCK_SIZE][BLOCK_SIZE];
 
        // Declaration of the local memory array Bs 
        // used to store the sub-matrix of B
        __local float Bs[BLOCK_SIZE][BLOCK_SIZE];
 
        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[lY][lX] = A[a + wA * lY + lX];
        Bs[lY][lX] = B[b + wB * lY + lX];
 
        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[lY][k] * Bs[k][lX];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * gY + BLOCK_SIZE * gX;
    C[c + wB * lY + lX] = Csub;


}