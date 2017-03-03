/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(float *reference, float *idata, size_t ntimepts, size_t width, size_t height);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! 
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float *reference, float *idata, size_t ntimepts, size_t width, size_t height)
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
			reference[y*width + x] = sum;
		}
	}

	return;
}

