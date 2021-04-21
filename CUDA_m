
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"

#include <stdio.h>
#include <omp.h>
#include <ctime>
#include <iostream>

const int NUM = 1025;

static double a[NUM][NUM], b[NUM][NUM], c[NUM][NUM], a_sq[NUM][NUM], a_sq_b[NUM][NUM], c_sq[NUM][NUM], res[NUM][NUM];

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void multiply_d(double a[][NUM], double b[][NUM], double c[][NUM])
{
	int i, j, k;
#pragma omp parallel for 
	for (i = 0; i < NUM; i++) {
		for (j = 0; j < NUM; j++) {
			//#pragma omp parallel for
			for (k = 0; k < NUM; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}
}

void sum_d(double a[][NUM], double b[][NUM], double c[][NUM]) {
	int i, j;
#pragma omp parallel for
	for (i = 0; i < NUM; i++)
	{
		for (j = 0; j < NUM; j++)
		{
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

__global__ void multiplyGPU(double* a, double* b, double *c)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	double sch = 0;
	if (tidx < NUM && tidy < NUM)
	{
		for (int i = 0; i < NUM; i++)
		{
			sch += a[tidx * NUM + i] * b[i * NUM + tidy];
		}
		c[tidx * NUM + tidy] = sch;
	}
}

__global__ void sumGPU(double* a, double* b, double* c)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidx < NUM && tidy < NUM)
		c[tidx * NUM + tidy] = a[tidx * NUM + tidy] + b[tidx * NUM + tidy];
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void init_arr(double row, double col, double off, double a[][NUM])
{
	int i, j;

	for (i = 0; i < NUM; i++)
	{
		for (j = 0; j < NUM; j++)
		{
			a[i][j] = (row * i + col * j + off) - (row * i + col * j + off) / 1 + 1;
			//a[i][j] = row * i + col * j + off;
		}
	}
}

void pprint(double a[][NUM]) {
	for (int i = 0; i < NUM; i++)
	{
		for (int j = 0; j < NUM; j++)
		{
			printf("%f ", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}



int main()
{

	double start, stop;

	init_arr(3, -2, 1, a);
	init_arr(-2, 1, 3, b);
	init_arr(-5, -2, 2, c);


	printf("NUM:%d\n", NUM);

	omp_set_num_threads(16);

	printf("Without MKL:");
	start = clock();
	multiply_d(a, a, a_sq);
	multiply_d(a_sq, b, a_sq_b);
	multiply_d(c, c, c_sq);
	sum_d(a_sq_b, c_sq, res);
	stop = clock();
	printf("Elapsed time without mkl = %lf seconds\n", ((double)(stop - start)) / 1000);


	double* a_plain = new double[NUM * NUM];
	double* b_plain = new double[NUM * NUM];
	double* ff = new double[NUM * NUM];
	double* c_plain = new double[NUM * NUM];

	double* a2, *b2, *c2, *r2_1, *r2_2, *r2_3, *r2_f;

	gpuErrchk(cudaMalloc((void**)&a2, NUM * NUM * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&b2, NUM * NUM * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&c2, NUM * NUM * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&r2_1, NUM * NUM * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&r2_2, NUM * NUM * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&r2_3, NUM * NUM * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&r2_f, NUM * NUM * sizeof(double)));


	for (int i = 0; i < NUM; i++)
	{
		for (int j = 0; j < NUM; j++)
		{
			a_plain[i * NUM + j] = a[i][j];
			b_plain[i * NUM + j] = b[i][j];
			c_plain[i * NUM + j] = c[i][j];
		}
	}

	gpuErrchk(cudaMemcpy(a2, a_plain, NUM * NUM * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(b2, b_plain, NUM * NUM * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(c2, c_plain, NUM * NUM * sizeof(double), cudaMemcpyHostToDevice));


	for (int j = 0; j < NUM; j++)
		for (int k = 0; k < NUM; k++)
		{
			a_sq[j][k] = 0;
			a_sq_b[j][k] = 0;
			c_sq[j][k] = 0;
			res[j][k] = 0;
		}

	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	dim3 gridSize = dim3((NUM + 31) / 32, (NUM + 31) / 32, 1);
	dim3 blockSize = dim3(32, 32, 1);

	cudaEventRecord(startGPU);

	multiplyGPU <<<gridSize, blockSize >>> (a2, a2, r2_1);
	multiplyGPU <<<gridSize, blockSize >>> (r2_1, b2, r2_2);
	multiplyGPU <<<gridSize, blockSize >>> (c2, c2, r2_3);
	sumGPU <<<gridSize, blockSize >>> (r2_2, r2_3, r2_f);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaEventRecord(stopGPU);
	gpuErrchk(cudaEventSynchronize(stopGPU));
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);

	gpuErrchk(cudaMemcpy(res, r2_f, NUM * NUM * sizeof(double), cudaMemcpyDeviceToHost));

	printf("CUDA: time: %f; ", (double)(milliseconds / 1000));
	printf("end");
	

	cudaFree(a2);
	cudaFree(b2);
	cudaFree(c2);
	cudaFree(r2_1);
	cudaFree(r2_2);
	cudaFree(r2_3);
	cudaFree(r2_f);
	system("pause");

	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };

	//// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "addWithCuda failed!");
	//    return 1;
	//}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//    c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "cudaDeviceReset failed!");
	//    return 1;
	//}

	//return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
