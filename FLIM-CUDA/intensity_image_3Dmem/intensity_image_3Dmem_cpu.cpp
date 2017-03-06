/* FLIM-CUDA-Primatives
 * P Barber <paul.barber@oncology.ox.ac.uk>, 2017
 * Sum the time data at each pixel to produce an intensity image
 * CPU Comparison code.
 */

// includes CUDA
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
/*! CPU Gold Standard to sum the time data at each pixel to produce an intensity image
 *
 * @param idata     input data
 * @param odata     output data
 * @param ntimepts  number of time points in a transient - the first dimension of the data
 * @param width     width of image - the second dimension of the data
 * @param height    height of image - the third dimension of the data
 */
extern "C" void
computeGold(cudaPitchedPtr *idata, float *odata, size_t ntimepts, size_t width, size_t height)
{

	size_t pixel_pitch = idata->pitch;  // in bytes
	size_t width_pitch = pixel_pitch * width;
	for (size_t y = 0; y < height; ++y)
    {
		char *row_start = (char*)idata->ptr + y * width_pitch;  // is char* because width_pitch is in bytes
		for (unsigned int x = 0; x < width; ++x)
		{
			float *trans = (float*)(row_start + x * pixel_pitch);
			float sum = 0.0;
			for (unsigned int t = 0; t < ntimepts; ++t)
			{
				sum += *trans;
				trans++;
			}
			odata[y*width + x] = sum;
		}
	}

	return;
}

