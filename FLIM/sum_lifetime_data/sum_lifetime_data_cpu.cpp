/* FLIM-CUDA-Primatives
 * P Barber <paul.barber@oncology.ox.ac.uk>, 2017
 * Sum the time data at each pixel to produce an intensity image
 * CPU Comparison code.
 */

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
computeGold(float *idata, float *odata, size_t ntimepts, size_t width, size_t height)
{

	return;
}

