#include<iostream>
#include<vector>

const int SHARED_MEM = 64;

__global__ void dotProdKernel(int *a, int *b, int *r, int N){
    __shared__ int sh[SHARED_MEM*sizeof(int)];
    int index = threadIdx.x + blockDim.x*blockIdx.x;

    int offset = 0;
    int stride = blockDim.x;
    
    while(index+offset < N){
        sh[threadIdx.x] += a[index+offset]*b[index+offset];
        offset += stride;
    }
    __syncthreads();

    int i = blockDim.x/2;
    while(i > 0){
        if(threadIdx.x < i){
            sh[threadIdx.x] += sh[threadIdx.x+i];
        }
        i /= 2;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        *r = sh[0];
    }

}


int main(){
    int N = 1024;
    int size = N*sizeof(int);

    std::vector<int> vec_a(N);
    std::vector<int> vec_b(N);
    int res = 0;

    int *d_a, *d_b, *d_res;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_res, sizeof(int));

    for(auto i = 0; i < N; i++){
        vec_a[i] = rand()%10;
        vec_b[i] = rand()%10;
    }

    cudaMemcpy(d_a, vec_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, vec_b.data(), size, cudaMemcpyHostToDevice);

    int ThreadsPerBlock = 64;
    int BlocksPerGrid = (N+ThreadsPerBlock-1)/ThreadsPerBlock;

    dotProdKernel<<<BlocksPerGrid,ThreadsPerBlock>>>(d_a, d_b, d_res, N);

    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Inner product is: " << res << std::endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);


    return 0;
}
