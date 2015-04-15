/**
 * gemm.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;
	
__kernel void gemm(__global DATA_TYPE *a, __global DATA_TYPE *b, __global DATA_TYPE *c, DATA_TYPE alpha, DATA_TYPE beta, int ni, int nj, int nk) 
{
	__local float sma[256]; // 16 x 16
	__local float smb[256];

	int bx = get_group_id(0); // col
	int by = get_group_id(1); // row

	int tx = get_local_id(0);
	int ty = get_local_id(1);


	int aBegin = nk * 16 * by;
	int aEnd = aBegin + nk - 1;
	int aStep = 16;

	int bBegin = 16 * bx;
	int bStep = 16 * nj;

	float sum = 0.f;

	int aa, bb;

	for(aa = aBegin, bb = bBegin; aa <= aEnd; aa += aStep, bb += bStep)
	{
		sma[ty * 16 + tx] = a[aa + ty * nk + tx];
		smb[ty * 16 + tx] = b[bb + ty * nj + tx];

		barrier(CLK_LOCAL_MEM_FENCE);

		
		int k;
		#pragma unroll
		for(k = 0; k < 16; ++k)
		{
			sum += sma[ty * 16 + k] * smb[k * 16 + tx];
		}
	
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	uint col = get_global_id(0);
	uint row = get_global_id(1);
	c[row * nj + col] = alpha * sum + beta * c[row * nj + col];
}
