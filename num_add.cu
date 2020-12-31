// Add 2 numbers

#include<iostream>

__global__ void numadd(float *d_a, float *d_b, float *d_c){
// int i = blockDim.x * blockIdx.x + threadIdx.x;
*d_c = *d_a + *d_b;

}

int main(){

float h_a, h_b, h_c;
std::cout << "Enter a number" << std::endl;
std::cin >> h_a;
std::cout << "Enter another number" << std::endl;
std::cin >> h_b;

size_t size = sizeof(float);
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, size);
cudaMalloc(&d_b, size);
cudaMalloc(&d_c, size);

cudaMemcpy(d_a, &h_a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, &h_b, size, cudaMemcpyHostToDevice);

int numBlocks = 1;
int threadsPerBlock = 1;
numadd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

cudaMemcpy(&h_c, d_c, size, cudaMemcpyDeviceToHost);

std::cout << "The sum is: "<< h_c << std::endl;


cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);

return 0;
}
