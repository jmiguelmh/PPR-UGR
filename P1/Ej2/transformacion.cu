#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

__global__ void transformacion_no_shared(float *d_A, float *d_B, float *d_C, float *d_D, float *d_mx)
{
    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;

    // Create pointer in shared memory
    extern __shared__ float sdata[];
    float *sdataA = sdata;
    float *sdataB = sdata + blockDim.x;
    float *sdataC = sdata + 2 * blockDim.x;
    float *sdataMX = sdata + 3 * blockDim.x;

    // From A and B to shared memory
    *(sdataA + tid) = d_A[i];
    *(sdataB + tid) = d_B[i];

    __syncthreads();

    // Compute C
    int istart = blockIdx.x * blockDim.x;
    int iend = istart + blockDim.x;

    for (int j = istart; j < iend; j++)
        d_C[i] += fabs((i * d_B[j] - d_A[j] * d_A[j]) / ((i + 2) * max(d_A[j], d_B[j])));

    *(sdataC + tid) = d_C[i];
    *(sdataMX + tid) = d_C[i];

    __syncthreads();

    // Compute D and mx
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdataC[tid] += sdataC[tid + s];
            *(sdataMX + tid) = *(sdataMX + tid);
            if (*(sdataMX + tid) < *(sdataMX + tid + s))
                *(sdataMX + tid) = *(sdataMX + tid + s);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_D[blockIdx.x] = *(sdataC) / blockDim.x;
        d_mx[blockIdx.x] = *(sdataMX);
    }
}

__global__ void transformacion_shared(float *d_A, float *d_B, float *d_C, float *d_D, float *d_mx)
{
    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;

    // Create pointer in shared memory
    extern __shared__ float sdata[];
    float *sdataA = sdata;
    float *sdataB = sdata + blockDim.x;
    float *sdataC = sdata + 2 * blockDim.x;
    float *sdataMX = sdata + 3 * blockDim.x;

    // From A and B to shared memory
    *(sdataA + tid) = d_A[i];
    *(sdataB + tid) = d_B[i];

    __syncthreads();

    // Compute C
    for (int j = 0; j < blockDim.x; j++)
        d_C[i] += fabs((i * sdataB[j] - sdataA[j] * sdataA[j]) / ((i + 2) * max(sdataA[j], sdataB[j])));

    *(sdataC + tid) = d_C[i];
    *(sdataMX + tid) = d_C[i];

    __syncthreads();

    // Compute D and mx
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdataC[tid] += sdataC[tid + s];
            *(sdataMX + tid) = *(sdataMX + tid);
            if (*(sdataMX + tid) < *(sdataMX + tid + s))
                *(sdataMX + tid) = *(sdataMX + tid + s);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_D[blockIdx.x] = *(sdataC) / blockDim.x;
        d_mx[blockIdx.x] = *(sdataMX);
    }
}

//**************************************************************************
int main(int argc, char *argv[])
//**************************************************************************
{
    int Bsize, NBlocks;
    if (argc != 3)
    {
        cout << "Uso: transformacion Num_bloques Tam_bloque  " << endl;
        return (0);
    }
    else
    {
        NBlocks = atoi(argv[1]);
        Bsize = atoi(argv[2]);
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

    const int N = Bsize * NBlocks;
    // pointers to host memory
    float *A, *B, *C, *D, *mx_no_shared, *D_shared, *mx_shared;

    // pointers to device memory 
    float *d_A, *d_B, *d_C, *d_D, *d_mx, *d_D_shared, *d_mx_shared;

    // Allocate arrays a, b and c on host
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[NBlocks];
    D_shared = new float[NBlocks];
    mx_no_shared = new float[NBlocks];
    mx_shared = new float[NBlocks];

    float mx; // maximum of C

    // Allocate device memory
    err = cudaMalloc((void **)&d_A, N * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "ERROR MALLOC IN d_A" << endl;
    }

    err = cudaMalloc((void **)&d_B, N * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "ERROR MALLOC IN d_B" << endl;
    }

    err = cudaMalloc((void **)&d_C, N * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "ERROR MALLOC IN d_C" << endl;
    }

    err = cudaMalloc((void **)&d_D, NBlocks * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "ERROR MALLOC IN d_D" << endl;
    }

    err = cudaMalloc((void **)&d_D_shared, NBlocks * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "ERROR MALLOC IN d_D_shared" << endl;
    }

    err = cudaMalloc((void **)&d_mx, NBlocks * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "ERROR MALLOC IN d_mx" << endl;
    }

    err = cudaMalloc((void **)&d_mx_shared, NBlocks * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "ERROR MALLOC IN d_mx_shared" << endl;
    }

    // Initialize arrays A and B
    for (int i = 0; i < N; i++)
    {
        A[i] = (float)(1.5 * (1 + (5 * i) % 7) / (1 + i % 5));
        B[i] = (float)(2.0 * (2 + i % 5) / (1 + i % 7));
    }

    /*********************** CPU Phase ***********************/
    // Time measurement
    double t1 = clock();

    // Compute C[i], D[k] and mx
    for (int k = 0; k < NBlocks; k++)
    {
        int istart = k * Bsize;
        int iend = istart + Bsize;
        for (int i = istart; i < iend; i++)
        {
            C[i] = 0.0;
            for (int j = istart; j < iend; j++)
                C[i] += fabs((i * B[j] - A[j] * A[j]) / ((i + 2) * max(A[j], B[j])));
        }
    }

    // Compute mx=max{Ci}
    mx = C[0];
    for (int i = 0; i < N; i++)
    {
        mx = max(C[i], mx);
    }

    // Compute D
    for (int k = 0; k < NBlocks; k++)
    {
        int istart = k * Bsize;
        int iend = istart + Bsize;
        D[k] = 0.0;
        for (int i = istart; i < iend; i++)
        {
            D[k] += C[i];
        }
        D[k] /= Bsize;
    }

    double t2 = clock();
    t2 = (t2 - t1) / CLOCKS_PER_SEC;

    /******************* GPU Phase (no-shared) *******************/
    double t1_gpu_no_shared = clock();

    // Copy from host to device memory
    err = cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN d_A" << endl;
    }

    err = cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN d_B" << endl;
    }

    // Kernel launch (shared memory is not used to compute C vector)
    int smemSize = 4 * Bsize * sizeof(float);
    transformacion_no_shared<<<NBlocks, Bsize, smemSize>>>(d_A, d_B, d_C, d_D, d_mx);

    // Copy from device to host memory
    err = cudaMemcpy(D, d_D, NBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN D" << endl;
    }

    err = cudaMemcpy(mx_no_shared, d_mx, NBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN mx_no_shared" << endl;
    }

    cudaDeviceSynchronize();

    // Final reduction on CPU (max)
    float mx_no_shared_result = mx_no_shared[0];
    for (int i = 0; i < NBlocks; i++)
        mx_no_shared_result = (mx_no_shared_result > mx_no_shared[i]) ? mx_no_shared_result : mx_no_shared[i];

    double t2_gpu_no_shared = clock();
    t2_gpu_no_shared = (t2_gpu_no_shared - t1_gpu_no_shared) / CLOCKS_PER_SEC;

    /******************** GPU Phase (shared) *********************/
    double t1_gpu_shared = clock();

    // Copy from host to device memory
    err = cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN d_A" << endl;
    }

    err = cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN d_B" << endl;
    }

    // Kernel launch (shared memory is used to compute C vector)
    smemSize = 4 * Bsize * sizeof(float);
    transformacion_shared<<<NBlocks, Bsize, smemSize>>>(d_A, d_B, d_C, d_D_shared, d_mx_shared);

    // Copy from device to host memory
    err = cudaMemcpy(D_shared, d_D_shared, NBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN D_shared" << endl;
    }

    err = cudaMemcpy(mx_shared, d_mx_shared, NBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << "ERROR CUDA MEM. COPY IN mx_shared" << endl;
    }

    cudaDeviceSynchronize();

    // Final reduction on CPU (max)
    float mx_shared_result = mx_shared[0];
    for (int i = 0; i < NBlocks; i++)
        mx_shared_result = (mx_shared_result > mx_shared[i]) ? mx_shared_result : mx_shared[i];

    double t2_gpu_shared = clock();
    t2_gpu_shared = (t2_gpu_shared - t1_gpu_shared) / CLOCKS_PER_SEC;

    /************************** Results **************************/
    cout << "CPU time:\t\t" << t2 << endl;
    cout << "GPU (no shared) time:\t" << t2_gpu_no_shared << endl;
    cout << "GPU (shared) time:\t" << t2_gpu_shared << endl;

    //* Free the memory */
    delete (A);
    delete (B);
    delete (C);
    delete (D);
    delete (D_shared);
    delete (mx_no_shared);
    delete (mx_shared);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_D_shared);
    cudaFree(d_mx);
    cudaFree(d_mx_shared);
}