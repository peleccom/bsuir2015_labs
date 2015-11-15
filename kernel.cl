
#define VALUE_TYPE float
#define BLOCK_SIZE 16
__kernel void cdot(const __global VALUE_TYPE* A,
                    int hA, int wA,
                      const __global VALUE_TYPE* B,
                      int hB, int wB,
                      __global VALUE_TYPE* C) {

__local VALUE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];
__local VALUE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

  int gY = get_group_id(0);
  int gX = get_group_id(1);
  int lY = get_local_id(0);
  int lX = get_local_id(1);
  int y = get_global_id(1);
  int x = get_global_id(0);


  VALUE_TYPE sum = 0;



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


 
        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        if (x < wA && y < hA){
          As[lY][lX] = A[a + wA * lY + lX];
          Bs[lY][lX] = B[b + wB * lY + lX];
        }
        else{
          As[lY][lX] = 0;
          Bs[lY][lX] = 0;
        }
        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += As[lY][k] * Bs[k][lX];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * gY + BLOCK_SIZE * gX;
    if (x < wA && y < hA)
      C[c + wB * lY + lX] = sum;

}