#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

using namespace std;

#define blocksize 256
#define blocksize2D 16
#define blocksizeReduction 256

//**************************************************************************

__global__ void floyd1D_kernel(int *M, const int nverts, const int k)
{
	int ij = threadIdx.x + blockDim.x * blockIdx.x;
	int i = ij / nverts;
	int j = ij - i * nverts;
	if (i < nverts && j < nverts)
	{
		int Mij = M[ij];
		if (i != j && i != k && j != k)
		{
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
		}
	}
}

__global__ void floyd2D_kernel(int *M, const int nverts, const int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < nverts && j < nverts)
	{
		int ij = i * nverts + j;
		int Mij = M[ij];

		if (i != j && i != k && j != k)
		{
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
		}
	}
}

__global__ void reductionKernel(int *d_In_M, int *d_Out_M, const int N)
{
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? d_In_M[i] : 0.0f);
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			sdata[tid] = (sdata[tid + s] < sdata[tid]) ? sdata[tid + s] : sdata[tid];

		__syncthreads();
	}

	if (tid == 0)
		d_Out_M[blockIdx.x] = sdata[0];
}

//**************************************************************************

//**************************************************************************
// ************  MAIN FUNCTION *********************************************
int main(int argc, char *argv[])
{

	double time, Tcpu, Tgpu1D, Tgpu2D;

	if (argc != 2)
	{
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return (-1);
	}

	// Get GPU information
	int num_devices, devID;
	cudaDeviceProp props;
	cudaError_t err;

	err = cudaGetDeviceCount(&num_devices);
	if (err == cudaSuccess)
	{
		cout << endl
			 << num_devices << " CUDA-enabled  GPUs detected in this computer system" << endl
			 << endl;
		cout << "....................................................." << endl
			 << endl;
	}
	else
	{
		cerr << "ERROR detecting CUDA devices......" << endl;
		exit(-1);
	}

	for (int i = 0; i < num_devices; i++)
	{
		devID = i;
		err = cudaGetDeviceProperties(&props, devID);
		cout << "Device " << devID << ": " << props.name << " with Compute Capability: " << props.major << "." << props.minor << endl
			 << endl;
		if (err != cudaSuccess)
		{
			cerr << "ERROR getting CUDA devices" << endl;
		}
	}
	devID = 0;
	cout << "Using Device " << devID << endl;
	cout << "....................................................." << endl
		 << endl;

	err = cudaSetDevice(devID);
	if (err != cudaSuccess)
	{
		cerr << "ERROR setting CUDA device" << devID << endl;
	}

	// Declaration of the Graph object
	Graph G;

	// Read the Graph
	G.lee(argv[1]);

	// cout << "The input Graph:"<<endl;
	// G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	int size = nverts2 * sizeof(int);
	int sizeReduction = ceil((float) nverts2 / blocksizeReduction);

	// Arrays in host memory
	int *c_Out_M_1D = new int[nverts2];
	int *c_Out_M_2D = new int[nverts2];
	int *c_Out_M_Reduction = new int[sizeReduction];

	// Arrays in device memory
	int *d_In_M_1D = NULL;
	int *d_In_M_2D = NULL;
	int *d_In_M_Reduction = NULL;
	int *d_Out_M_Reduction = NULL;
	
	// Allocate device memory
	err = cudaMalloc((void **)&d_In_M_1D, size);
	if (err != cudaSuccess)
	{
		cerr << "ERROR MALLOC IN d_in_M_1D" << endl;
	}

	err = cudaMalloc((void **)&d_In_M_2D, size);
	if (err != cudaSuccess)
	{
		cerr << "ERROR MALLOC IN d_in_M_2D" << endl;
	}

	err = cudaMalloc((void **)&d_In_M_Reduction, size);
	if (err != cudaSuccess)
	{
		cerr << "ERROR MALLOC IN d_In_M_Reduction" << endl;
	}

	err = cudaMalloc((void **)&d_Out_M_Reduction, sizeReduction*sizeof(int));
	if (err != cudaSuccess)
	{
		cerr << "ERROR MALLOC IN d_Out_M_Reduction" << endl;
	}

	// Get the integer 2D array for the dense graph
	int *A = G.Get_Matrix();

	/*********************** CPU Phase ***********************/

	time = clock();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for (int k = 0; k < niters; k++)
	{
		kn = k * nverts;
		for (int i = 0; i < nverts; i++)
		{
			in = i * nverts;
			for (int j = 0; j < nverts; j++)
				if (i != j && i != k && j != k)
				{
					inj = in + j;
					A[inj] = min(A[in + k] + A[kn + j], A[inj]);
				}
		}
	}

	Tcpu=(clock()-time)/CLOCKS_PER_SEC;

	/*********************** GPU Phase 1D ************************/
	
	time=clock();

	err = cudaMemcpy(d_In_M_1D, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY IN d_In_M_1D" << endl;
	} 

    // Main Loop
	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	int threadsPerBlock = blocksize;
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
        // Kernel Launch
	    floyd1D_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M_1D, nverts, k);
	    err = cudaGetLastError();

	    if (err != cudaSuccess) {
	  	    fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	    exit(EXIT_FAILURE);
		}
	}
	err =cudaMemcpy(c_Out_M_1D, d_In_M_1D, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY IN c_Out_M_1D" << endl;
	} 

	Tgpu1D=(clock()-time)/CLOCKS_PER_SEC;

	/*********************** GPU Phase 2D ************************/

	time=clock();

	err = cudaMemcpy(d_In_M_2D, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY IN d_In_M_2D" << endl;
	} 

    // Main Loop
	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	dim3 threadsPerBlock(blocksize2D, blocksize2D);
	 	dim3 blocksPerGrid(ceil((float) nverts/blocksize2D), ceil((float) nverts/blocksize2D));
        // Kernel Launch
	    floyd2D_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M_2D, nverts, k);
	    err = cudaGetLastError();

	    if (err != cudaSuccess) {
	  	    fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	    exit(EXIT_FAILURE);
		}
	}
	err =cudaMemcpy(c_Out_M_2D, d_In_M_2D, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY IN c_Out_M_2D" << endl;
	} 

	Tgpu2D=(clock()-time)/CLOCKS_PER_SEC;

	/************************** Results **************************/
    cout << "CPU time:\t\t" << Tcpu << endl;
    cout << "GPU (1D) time:\t" << Tgpu1D << endl;
    cout << "GPU (2D) time:\t" << Tgpu2D << endl;

	/******************* GPU Reduction Phase *********************/
	
	err = cudaMemcpy(d_In_M_Reduction, c_Out_M_2D, sizeReduction * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cout << "ERROR CUDA MEM. COPY IN d_In_M_Reduction" << endl;
	}

	int smemSize = sizeReduction;
	reductionKernel<<<sizeReduction, blocksizeReduction, smemSize>>>(d_In_M_Reduction, d_Out_M_Reduction, nverts2);
	cudaDeviceSynchronize();


	// Copy from device to host
	cudaMemcpy(c_Out_M_Reduction, d_Out_M_Reduction, sizeReduction * sizeof(int), cudaMemcpyDeviceToHost);

	// Final reduction on CPU (compute average value for shortest paths)
	double average = 0.0;
	for(int i = 0; i < sizeReduction; i++)
		average += c_Out_M_Reduction[i];

	average /= sizeReduction;
	/*

	bool errors = false;
	// Error Checking (CPU vs. GPU)
	for (int i = 0; i < nverts; i++)
		for (int j = 0; j < nverts; j++)
			if (abs(c_Out_M[i * nverts + j] - G.arista(i, j)) > 0)
			{
				cout << "Error (" << i << "," << j << ")   " << c_Out_M[i * nverts + j] << "..." << G.arista(i, j) << endl;
				errors = true;
			}

	if (!errors)
	{
		cout << "....................................................." << endl;
		cout << "WELL DONE!!! No errors found ............................" << endl;
		cout << "....................................................." << endl
			 << endl;
	}
	*/
	
	// Free host memory
	delete(c_Out_M_1D);
	delete(c_Out_M_2D);
	delete(c_Out_M_Reduction);

	// Free device memory
	cudaFree(d_In_M_1D);
	cudaFree(d_In_M_2D);
	cudaFree(d_In_M_Reduction);
	cudaFree(d_Out_M_Reduction);
}
