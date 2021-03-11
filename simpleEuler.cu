#include<iostream>
#include<vector>

__device__ void derivative(float &x, float &y){y = 2*x;}

__global__ void EulerKernel(float *x, float *y, float *res, int N, int n_itr, float del_t, float del_x){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	float der_value;
	
	for(int it = 0; it < n_itr; it++){
		derivative(x[index], der_value);
		// y[index] += (del_t/del_x)*(2*x[index]);
		y[index] += (del_t/del_x)*(der_value);
	}
	__syncthreads();
	
	res[index] = y[index];
	// printf("%f\n", res[index]);

}

int main(){
	int N = 1024;
	int size = N*sizeof(float);
	
	int num_itr = 1;
	
	std::vector<float> x(N,0.0);
	std::vector<float> y(N,0.0);
	std::vector<float> res(N,0.0);
	
	for(auto i = 0; i<N; i++){
		x[i] = i;
		y[i] = i*i;
	}
	float del_t = 0.001;
	float del_x = x[1] - x[0];	//  assuming costant del_x
	
	float *d_x, *d_y, *d_res;
	// float *d_del_t, *d_del_x;
	// int *d_itr;
	cudaMalloc((void **)&d_x, size);
	cudaMalloc((void **)&d_y, size);
	cudaMalloc((void **)&d_res, size);
	// cudaMalloc((void **)&d_del_t, sizeof(float));
	// cudaMalloc((void **)&d_del_x, sizeof(float));
	// cudaMalloc((void **)&d_itr, sizeof(int));
	
	cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_del_t, del_t, sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_del_x, del_x, sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_itr, num_itr, sizeof(int), cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 32;
	int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
	EulerKernel<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_res,N,num_itr,del_t,del_x);
	
	cudaMemcpy(res.data(), d_res, size, cudaMemcpyDeviceToHost);
	
	for(const auto& i:res){std::cout << i << std::endl;}
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_res);

	return 0;
}
