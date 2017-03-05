/* FLIM-CUDA-Primatives
 * P Barber <paul.barber@oncology.ox.ac.uk>, 2017
 * Sum the time data at each pixel to produce an intensity image
 * GPU code.
 */

// includes CUDA
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
/*! Kernel to sum the time data at each pixel to produce an intensity image
 *
 * @param g_idata   input data in global memory
 * @param g_odata   output data in global memory
 * @param ntimepts  number of time points in a transient - the first dimension of the data
 * @param width     width of image - the second dimension of the data
 * @param height    height of image - the third dimension of the data
 */
__global__ void
CudaKernel(float *g_idata, float *g_odata, size_t ntimepts, size_t width, size_t height)
{
	// we should create one thread per pixel, and one thread block per row of the image
	int y = blockIdx.x;    // The block indicates the row
    int x = threadIdx.x;   // The thread in the block indicates the col
	
	// TODO - Code here :-)
	// This may have one thread per entry in the output array (256), and each one strides through the data in global mem adding them together
	// OR it may have one thread per pixel, and each one takes its lifetime data and adds it to the output array in shared memory, need to use atomic adds.
}

////////////////////////////////////////////////////////////////////////////////
/*! Wrapper function to run the kernel from cpu c code.
 */
extern "C" void 
executeCudaKernel(unsigned int threadsPerBlock, unsigned int  blocksPerGrid, float *d_idata, float *d_odata, size_t ntimepts, size_t width, size_t height)
{
	CudaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, ntimepts, width, height);
}