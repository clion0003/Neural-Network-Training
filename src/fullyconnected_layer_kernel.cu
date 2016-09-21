#ifdef GPU
#include "fullyconnected_layer.h"


__global__ void sigmoidkernel(float *a, int max_index)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<max_index)a[i] = 1.0 / (1.0 + exp(-a[i]));
}

__global__ void dsigmoidKernel(float *g_, float *input, int max_index)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < max_index)g_[i] = g_[i] * (input[i] * (1 - input[i]));
}

__global__ void vectordotKernel(float *dst, float *a, float *b, int max_index)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < max_index)dst[i] = a[i] * b[i];
}

__global__ void fixkernel(float *dst, char *fix,int max_index)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < max_index && fix[i]==1)dst[i] = 0;
}


namespace mlp{

	void FullyConnectedLayer::transfer_weight_h2d(){
		checkerror(cudaMemcpy(W_gpu, W_.data(), in_depth_*out_depth_*sizeof(float),cudaMemcpyHostToDevice), "transfer W_ h2d");
		//checkerror(cudaMemcpy(W_gpu, W_.data(), in_depth_*out_depth_*sizeof(float), cudaMemcpyHostToDevice), "transfer W_ h2d");
		checkerror(cudaMemcpy(b_gpu, b_.data(), out_depth_*sizeof(float), cudaMemcpyHostToDevice), "transfer b_ h2d");
	}

	void FullyConnectedLayer::transfer_weight_d2h(){
		checkerror(cudaMemcpy(W_.data(), W_gpu, in_depth_*out_depth_*sizeof(float), cudaMemcpyDeviceToHost), "transfer W_ d2h");
		//checkerror(cudaMemcpy(W_gpu, W_.data(), in_depth_*out_depth_*sizeof(float), cudaMemcpyHostToDevice), "transfer W_ h2d");
		checkerror(cudaMemcpy(b_.data(), b_gpu, out_depth_*sizeof(float), cudaMemcpyDeviceToHost), "transfer b_ d2h");
	}

	void FullyConnectedLayer::transfer_deltaW_h2d(){
		checkerror(cudaMemcpy(deltaW_gpu, deltaW_.data(), out_depth_*in_depth_*sizeof(float),cudaMemcpyHostToDevice), "transfer delta_W h2d");
	}

	void FullyConnectedLayer::transfer_input_h2d(){
		checkerror(cudaMemcpy(input_gpu, input_.data(), in_depth_*sizeof(float), cudaMemcpyHostToDevice), "transfer input_ h2d");
	}


	void FullyConnectedLayer::transfer_fault_h2d(){
		float *fault;
		fault = new float[(in_depth_ + 1)*out_depth_];
		for (size_t i = 0; i < out_depth_; i++)
		{
			for (size_t j = 0; j < in_depth_ + 1; j++)
			{
				fault[i*(in_depth_ + 1) + j] = (float)Fault_.getFaultValue(i, j);
			}
		}
		checkerror(cudaMemcpy(fault_gpu, fault, in_depth_*out_depth_*sizeof(float), cudaMemcpyHostToDevice), "transfer fault h2d");
		free(fault);
	}

	void FullyConnectedLayer::transfer_fixed_h2d(){
		checkerror(cudaMemcpy(W_fix_gpu, W_fix.data(), in_depth_*out_depth_*sizeof(char), cudaMemcpyHostToDevice), "transfer W_fix h2d");
		checkerror(cudaMemcpy(b_fix_gpu, b_fix.data(), out_depth_*sizeof(char), cudaMemcpyHostToDevice), "transfer b_fix h2d");
	}
	
	void FullyConnectedLayer::transfer_output_d2h(){
		checkerror(cudaMemcpy(output_.data(), output_gpu, out_depth_*sizeof(float), cudaMemcpyDeviceToHost), "transfer output d2h");
	}

	void FullyConnectedLayer::forward_gpu(){
		const float alpha = 1.0f;
		const float beta = 1.0f;
		checkerror(cudaMemcpy(output_gpu, b_gpu, out_depth_*sizeof(float), cudaMemcpyDeviceToDevice), "memcpy b_->output");
		cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, out_depth_, 1, in_depth_, &alpha, W_gpu, in_depth_, input_gpu, in_depth_, &beta, output_gpu, 10);
		dim3 thread_num(32);
		dim3 block_num((out_depth_+31)/32);
		sigmoidkernel << < block_num, thread_num >> > (output_gpu, out_depth_);
		checkerror(cudaDeviceSynchronize(),"sigmoid kernel");
	}

	void FullyConnectedLayer::forward_fault_gpu()
	{

	}

	void FullyConnectedLayer::fix_backprop_gpu(){
		const float alpha = 1.0f;
		const float beta = 0.0f;
		cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, in_depth_, out_depth_, &alpha, this->next->g_gpu, 1, W_gpu, in_depth_, &beta, g_gpu, 1);
		dim3 thread_num(32);
		dim3 block_num((in_depth_ + 31) / 32);
		dsigmoidKernel << < block_num, thread_num >> > (g_gpu, input_gpu, in_depth_);
		checkerror(cudaDeviceSynchronize(), "dx sigmoid kernel");
		cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, in_depth_, out_depth_, 1, &alpha_, input_gpu, in_depth_, this->next->g_gpu, 1, &lambda_, deltaW_gpu, in_depth_);
		
		dim3 thread_num2(32);
		dim3 block_num2((in_depth_*out_depth_ + 31) / 32);
		fixkernel << <block_num2, thread_num2 >> >(deltaW_gpu, W_fix_gpu, in_depth_*out_depth_);
		checkerror(cudaDeviceSynchronize(), "fix W kernel");

		dim3 thread_num3(32);
		dim3 block_num3((out_depth_ + 31) / 32);
		fixkernel << <block_num3, thread_num3 >> >(this->next->g_gpu, b_fix_gpu, out_depth_);
		checkerror(cudaDeviceSynchronize(), "fix b kernel");

		cublasSaxpy(cublas_handle, in_depth_*out_depth_, &alpha, deltaW_gpu, 1, W_gpu, 1);
		cublasSaxpy(cublas_handle, out_depth_, &alpha, this->next->g_gpu, 1, b_gpu, 1);
	}

}
#endif
