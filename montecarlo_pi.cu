#include<iostream>
#include<vector>
#include<curand.h>
#include<curand_kernel.h>

const int SHARED_MEM = 256;

__global__ void piEstimate(double *result, int N){
auto index = blockDim.x*blockIdx.x+threadIdx.x;
auto stride = blockDim.x;
auto offset = 0;

__shared__ double shared_mem[SHARED_MEM];
shared_mem[threadIdx.x]= 0;	// initializing to 0
__syncthreads();

curandState_t state;
while(index+offset < N){
curand_init(123456789,0,0,&state);
double x = curand_normal(&state);
double y = curand_normal(&state);

if(x*x + y*y <= 1){
shared_mem[threadIdx.x]++;
}

offset += stride;
}
// __syncthreads();

// Reduction
int i = blockIdx.x/2;
while(i > 0){
shared_mem[threadIdx.x] += shared_mem[threadIdx.x+i];
i/=2;
}
__syncthreads();

// First element is the output;
if(threadIdx.x == 0){
*result = shared_mem[0];
}


}

int main(){
auto N = 10240;
double *h_pi;
h_pi = (double*)malloc(sizeof(double));

double *d_pi;
cudaMalloc(&d_pi, sizeof(double));

auto threadsPerBlock = 32;
auto blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
piEstimate<<<blocksPerGrid, threadsPerBlock>>>(d_pi, N);

cudaMemcpy(h_pi, d_pi, sizeof(double), cudaMemcpyDeviceToHost);

std::cout << "Value of pi is: " << *h_pi*100/N<< std::endl;

free(h_pi);
cudaFree(d_pi);

return 0;
}
