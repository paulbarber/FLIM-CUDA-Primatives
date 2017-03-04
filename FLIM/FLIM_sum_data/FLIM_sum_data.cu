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
 * @param ntimepts  number of time points in a transient - the first dimention of the data
 * @param width     width of image - the second dimention of the data
 * @param height    height of image - the third dimention of the data
 */
__global__ void
CudaKernel(float *g_idata, float *g_odata, size_t ntimepts, size_t width, size_t height)
{
	// we should create one thread per pixel, and one thread block per row of the image
	int y = blockIdx.x;    // The block indicates the row
    int x = threadIdx.x;   // The thread in the block indicates the col
	
	//size_t pixel_pitch = g_idata->pitch;
	//size_t width_pitch = pixel_pitch * width;
	//char *row_start = (char*)g_idata->ptr + y * width_pitch;  // is char* because width_pitch is in bytes
	//float *trans = (float*)(row_start + x * pixel_pitch);

	float *row_start = g_idata + y * width;
	float *trans = row_start + x;

	float sum = 0.0;
	for(size_t t=0; t<ntimepts; ++t){
		sum += *trans;
		trans++;
	}
	
	g_odata[y*width + x] = sum;
}

////////////////////////////////////////////////////////////////////////////////
/*! Wrapper function to run the kernel from cpu c code.
 */
extern "C" void 
executeCudaKernel(unsigned int threadsPerBlock, unsigned int  blocksPerGrid, float *d_idata, float *d_odata, size_t ntimepts, size_t width, size_t height)
{
	CudaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, ntimepts, width, height);
}