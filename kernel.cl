
#define VALUE_TYPE float
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
  int y = get_global_id(0);
  int x = get_global_id(1);

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


 
        int bInd = b + wB * lY + lX;
        int aInd = a + wA * lY + lX;
        if (aInd < (wA * hA)){
          As[lY][lX] = A[aInd];
        }
        else{
          As[lY][lX] = 0;
        }
        if (bInd < (wB * hB)){
          Bs[lY][lX] = B[bInd];
        }
        else{
          Bs[lY][lX] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < BLOCK_SIZE; ++k){
            sum += As[lY][k] * Bs[k][lX];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
    int c = wB * BLOCK_SIZE * gY + BLOCK_SIZE * gX;
    if (x < wA && y < hA)
      C[c + wB * lY + lX] = sum;

}