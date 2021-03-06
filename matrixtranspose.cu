#include<iostream>

__global__ void transKernel(int *A, int *A_t, int N){
	int x_index = threadIdx.x + blockIdx.x*blockDim.x;
	int y_index = threadIdx.y + blockIdx.y*blockDim.y;
	
	if(x_index < N && y_index < N){
		A_t[x_index*N+y_index] = A[y_index*N+x_index];
	}
}

int main(){
	int N = 256;
	int size = N*N*sizeof(int);
	
	int A[N][N], At[N][N];
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			A[i][j] = rand()%10;
		}
	}
	
	int *d_A, *d_At;
	
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_At, size);
	
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock (8,8);
	dim3 blocksPerGrid (N/8, N/8);
	transKernel<<<threadsPerBlock, blocksPerGrid>>>(d_A, d_At, N);
	
	cudaMemcpy(At, d_At, size, cudaMemcpyDeviceToHost);
	
	// for(int i = 0; i < N; i++){
	// 	for(int j = 0; j < N; j++){
	// 		std::cout << At[i][j] << std::endl;
	// 	}
	// }
	
	cudaFree(d_A);
	cudaFree(d_At);

	return 0;
}
