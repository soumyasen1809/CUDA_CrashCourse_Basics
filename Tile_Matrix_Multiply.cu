#include<iostream>
#include<vector>

const int N = 16*16;
const int sharedMemsize = 16*16*sizeof(float);

__global__ void matMultiply(float *A, float *B, float *C){
	auto i = blockDim.y * blockIdx.y + threadIdx.y;
	auto j = blockDim.x * blockIdx.x + threadIdx.x;
	
	__shared__ float s_A[sharedMemsize];
	__shared__ float s_B[sharedMemsize];
	
	float temp = 0;
	for(auto k = 0; k < N; k++){
		s_A[threadIdx.y*blockDim.x+threadIdx.x] = A[i*N+k+threadIdx.x];
		s_B[threadIdx.y*blockDim.x+threadIdx.x] = B[(k+threadIdx.y)*N+j];
		__syncthreads();
		
		for(auto l = 0; l < blockDim.x; l++){
			temp += s_A[threadIdx.y*blockDim.x+l]*s_B[l*blockDim.x+threadIdx.x];
		}
		__syncthreads();
	}
	C[i*N+j] = temp;
}


int main(){
std::vector<float> h_A(N*N, 0);
std::vector<float> h_B(N*N, 0);
std::vector<float> h_C(N*N, 0);

for(auto i = 0; i < N*N; i++){
    h_A[i] = 0.01*i;
    h_B[i] = 0.002*i;
}

size_t size = N*N*sizeof(float);
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, size);
cudaMalloc(&d_B, size);
cudaMalloc(&d_C, size);

cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

// int thread = 1024;
// int blocks = N/1024;
dim3 ThreadsPerBlock(32, 32);		// 32 threads per row and per column
dim3 BlocksPerGrid(N/32,N/32);
matMultiply<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C);

cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

// Uncomment to see the output
// for(auto& i:h_C){
// 	std::cout << i << std::endl;
// }


cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

return 0;
}
