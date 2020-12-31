#include<iostream>
#include<cuda.h>

// Device code

__global__ void VecAdd(float* A, float* B, float* C, int N){
    	int i = blockDim.x * blockIdx.x + threadIdx.x;
    
	    if(i < N){
	    	C[i] = A[i] + B[i];
    	}
}

// Host code

int main(){
	int N = 10;
	size_t size = N*sizeof(float);
	
	// Allocate memory for the host
	// float* h_A = (float*)malloc(size);
	// float* h_B = (float*)malloc(size);
	// float* h_C = (float*)malloc(size);
	
	
	// Another way of writing - Pinned memory
	float *h_A, *h_B, *h_C;
	cudaMallocHost(&h_A, size);
	cudaMallocHost(&h_B, size);
	cudaMallocHost(&h_C, size);
	
	// Initialize the input vectors
	for(auto i = 0; i < N; i++){
		h_A[i] = i;
		h_B[i] = 2*i;
		h_C[i] = 0;
	}
	
	// Allocate memory for the device
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	
	// Copy contents of host to device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
	
	// Invoke the kernel to do the computation
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	
	// Copy results from device to host
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	// Free the memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	// Print values
	for(auto i = 0; i < N; i++){
		std::cout << h_C[i] << std::endl;
	}
	
	// Free Pinned memory
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);

	return 0;
}
