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
 * @param ntimepts  number of time points in a transient - the first dimention of the data
 * @param width     width of image - the second dimention of the data
 * @param height    height of image - the third dimention of the data
 */
extern "C" void
computeGold(float *idata, float *odata, size_t ntimepts, size_t width, size_t height)
{

	for(int y=0; y<height; ++y){
		float *row_start = idata + y * width;
		for(int x=0; x<width; ++x){
			float *trans = row_start + x;
			float sum = 0.0;
			for(size_t t=0; t<ntimepts; ++t){
				sum += *trans;
				trans++;
			}
			odata[y*width + x] = sum;
		}
	}

	return;
}

