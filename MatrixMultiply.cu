#include<iostream>
#include<vector>

__global__ void matMultiply(float *A, float *B, float *C, int N){
	auto i = blockDim.y * blockIdx.y + threadIdx.y;
	auto j = blockDim.x * blockIdx.x + threadIdx.x;
		
	// C[i*N+j] = 0.0;
	float temp = 0;
	for (int k = 0; k < N; k++){
		temp += A[i*N+k]*B[k*N+j];
	}
	C[i*N+j] = temp;
}


int main(){
int N = 32;	// NOTE: This size has to be >= 32, becasue we are using 32 threads per row. Less than 32 shall give the ouput vector as 0
std::vector<float> h_A(N*N, 0);
std::vector<float> h_B(N*N, 0);
std::vector<float> h_C(N*N, 0);

for(auto i = 0; i < N*N; i++){
    h_A[i] = 2*i;
    h_B[i] = 3*i;
}

size_t size = N*N*sizeof(float);
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, size);
cudaMalloc(&d_B, size);
cudaMalloc(&d_C, size);

cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

// int thread = 32;
// int blocks = N/32;
dim3 ThreadsPerBlock(32, 32);		// 32 threads per row and per column
dim3 BlocksPerGrid(N/32,N/32);
matMultiply<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, N);

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
