/* FLIM-CUDA-Primatives
 * P Barber <paul.barber@oncology.ox.ac.uk>, 2017
 * Sum the time data at each pixel to produce an intensity image
 * Main CPU code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declarations, forward
extern "C"
void computeGold(float *reference, float *idata, size_t ntimepts, size_t width, size_t height);

extern "C" 
void executeCudaKernel(size_t threadsPerBlock, size_t blocksPerGrid, float *g_idata, float *g_odata, size_t ntimepts, size_t width, size_t height);

////////////////////////////////////////////////////////////////////////////////
// Macro to time CPU operations
clock_t time_start, time_msec, time_diff;
#define time_this(a) time_start=clock(); a; time_diff = clock() - time_start; time_msec = time_diff * 1000 / CLOCKS_PER_SEC; printf("%s: %d ms\n", #a, time_msec);

////////////////////////////////////////////////////////////////////////////////
//! Test code to run the Kernel
int
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

	// setup some CUDA timers
	StopWatchInterface *hostToDeviceTimer = 0;
	StopWatchInterface *timer = 0;
	StopWatchInterface *DeviceToHostTimer = 0;
    sdkCreateTimer(&hostToDeviceTimer);
    sdkCreateTimer(&timer);
    sdkCreateTimer(&DeviceToHostTimer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    size_t width = 128;
    size_t height = 128;
    size_t ntimepts = 256;
	size_t mem_size = height * width * ntimepts * sizeof(float);
	size_t image_size = height * width * sizeof(float);
	
    // allocate host memory
	// data from ics file will be t,x,y (contigious data is t - height (y) has largest stride)
	// keep 3d data format the same (in cuda 3D contigious data is w - then h - d=depth has the largest stride)
//	cudaPitchedPtr h_idata = make_cudaPitchedPtr(malloc(mem_size), ntimepts*sizeof(float), ntimepts, width); 
	float *h_idata = (float*)malloc(mem_size); 

    // initalize the memory, using the recommend way to access cudaPitchedPtr data
//	size_t pixel_pitch = h_idata.pitch;
//	size_t width_pitch = pixel_pitch * width;
	for (size_t y = 0; y < height; ++y)
    {
//		char *row_start = (char*)h_idata.ptr + y * width_pitch;  // is char* because width_pitch is in bytes
		float *row_start = h_idata + y * width * ntimepts;
		for (unsigned int x = 0; x < width; ++x)
		{
//			float *time = (float*)(row_start + x * pixel_pitch);
			float *time = row_start + x * ntimepts;
			for (unsigned int t = 0; t < ntimepts; ++t)
			{
				time[t] = (float)(255-t);
			}
		}
	}

    sdkStartTimer(&hostToDeviceTimer);

    // allocate device memory
//    cudaPitchedPtr d_idata;
//    cudaExtent extent = make_cudaExtent(ntimepts*sizeof(float), width, height);
//    checkCudaErrors(cudaMalloc3D(&d_idata, extent));
	float *d_idata;
	checkCudaErrors(cudaMalloc(&d_idata, mem_size));

	// copy host memory to device
	//cudaMemcpy3DParms memcpy3DParms = {0};
	//memcpy3DParms.srcPtr = h_idata;
	//memcpy3DParms.dstPtr = d_idata;
	//memcpy3DParms.kind = cudaMemcpyHostToDevice;
	//memcpy3DParms.extent = extent;
	////memcpy3DParms.srcPos = make_cudaPos(0,0,0);
	////memcpy3DParms.dstPos = make_cudaPos(0,0,0);
 //   checkCudaErrors(cudaMemcpy3D(&memcpy3DParms));
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

	// allocate device memory for result
    float *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, image_size));

    sdkStopTimer(&hostToDeviceTimer);
    sdkStartTimer(&timer);

    // setup execution parameters
	unsigned int  threadsPerBlock = (unsigned int)width;
    unsigned int  blocksPerGrid = (unsigned int)(width*height) / threadsPerBlock;
	executeCudaKernel(blocksPerGrid, threadsPerBlock, d_idata, d_odata, ntimepts, width, height);

    sdkStopTimer(&timer);
    sdkStartTimer(&DeviceToHostTimer);
	
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(image_size);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, image_size, cudaMemcpyDeviceToHost));

    sdkStopTimer(&DeviceToHostTimer);

    // Report timings
	printf("GPU:\n");
	printf("Host To Device time: %f (ms)\n", sdkGetTimerValue(&hostToDeviceTimer));
	printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
	printf("Device To Host time: %f (ms)\n", sdkGetTimerValue(&DeviceToHostTimer));
    sdkDeleteTimer(&hostToDeviceTimer);
    sdkDeleteTimer(&timer);
    sdkDeleteTimer(&DeviceToHostTimer);

    // compute reference solution
    float *reference = (float *) malloc(image_size);
	printf("CPU:\n");
	time_this(computeGold(h_idata, reference, ntimepts, width, height));

    // check result
    //if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    //{
    //    // write file for regression test
    //    sdkWriteFile("./data/regression.dat", h_odata, num_threads, 0.0f, false);
    //}
    //else
    //{
    //    // custom output handling when no regression test running
    //    // in this case check if the result is equivalent to the expected soluion
    //    bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);
    //}

    // cleanup memory
	free(h_idata);
    free(h_odata);
    free(reference);
	checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits   
    cudaDeviceReset();
	
	return(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Program main
int
main(int argc, char **argv)
{
    int ret = runTest(argc, argv);
 
	printf("Press a key to finish.\n");
	getc(stdin);

	exit(ret);
}
