#include<iostream>
#include<vector>
#include<cstdlib>

const int M = 7;	// size of the mask
__constant__ double mask[M];

__global__ void convolution_kernel(double *arr, double *output, int N){
	auto i = blockDim.x*blockIdx.x+threadIdx.x;
	extern __shared__ double sharedArray[];	// use extern keyword because we don't know the size of the shared array; extern means dynamically allocated
	
	// Load the elements
	sharedArray[threadIdx.x] = arr[i];
	
	auto temp = 0.0;
	for(auto k = 0; k < M; k++){
		if(threadIdx.x+k >= blockDim.x){
			temp += arr[i+k]*mask[k];
		}
		else{
			temp += sharedArray[threadIdx.x+k]*mask[k];
		}
	}
	output[i] = temp;
}

int main(){

int N = 1048576;	// size of the array = 2^20
// int N = 1024;
int Npad = N + M;	// size of the padding
size_t size_N = N*sizeof(double);
size_t size_M = M*sizeof(double);
size_t size_Npad = Npad*sizeof(double);

std::vector<double> h_array(Npad);
std::vector<double> h_mask(M);
std::vector<double> h_output(N);

for(auto i = 0; i < Npad; i++){
	if((i < M/2) || (i >= N+(M/2))){
		h_array[i] = 0;
	}
	else{
		h_array[i] = rand()%100;
	}
}
for(auto& j:h_mask){j = rand()%10;}

double *d_array, *d_output;
cudaMalloc(&d_array, size_Npad);
cudaMalloc(&d_output, size_N);

cudaMemcpy(d_array, h_array.data(), size_Npad, cudaMemcpyHostToDevice);
cudaMemcpyToSymbol(mask, h_mask.data(), size_M);

int threadsPerBlock = 256;
int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
size_t sharedMem = (threadsPerBlock+M)*sizeof(double);	// number of threads + the halo (size M/2) on the left and right ends
convolution_kernel<<<blocksPerGrid, threadsPerBlock, sharedMem>>>(d_array, d_output, N);

cudaMemcpy(h_output.data(), d_output, size_N, cudaMemcpyDeviceToHost);

// Uncomment to print the output
// for(auto& i:h_output){std::cout << i << std::endl;}

cudaFree(d_array);
cudaFree(d_output);

return 0;
}


