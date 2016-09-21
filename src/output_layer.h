#ifndef OUTPUT_LAYER_H_
#define OUTPUT_LAYER_H_

#pragma once

#include "layer.h"
#include "util.h"

namespace mlp{
	class OutputLayer :public Layer
	{
	public:
		OutputLayer(size_t in_depth, activation* a) :
			Layer(in_depth, 0, a)
		{
#ifdef GPU
			if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS){
				std::cout << "cublas handle create failed!" << std::endl;
			}
			checkerror(cudaMalloc(&g_gpu, in_depth_*sizeof(float)), "malloc g_gpu");
			checkerror(cudaMalloc(&exp_y_gpu, in_depth_*sizeof(float)), "malloc g_gpu");
#endif		
		}

		void forward();

		void back_prop();

		void fix_backprop();

		void init_weight(){};

		void fault_forward();

		void generateFault_varition(float s){};

		void generateFault_sa(){};

		void find_fixed(int number){};

		void remap_best(){};

#ifdef GPU
		void transfer_weight_h2d(){};
		void transfer_weight_d2h(){};
		void transfer_input_h2d(){};
		void transfer_output_d2h(){};
		void init_cuda_memory(){};
		void forward_gpu();
		void forward_fault_gpu(){};
		void fix_backprop_gpu();
		void transfer_fault_h2d(){};
		void transfer_fixed_h2d(){};
		void transfer_deltaW_h2d(){};
#endif

	private:
		~OutputLayer(){
#ifdef GPU
			checkerror(cudaFree(g_gpu), "free g_gpu");
			checkerror(cudaFree(exp_y_gpu), "freee g_gpu");
#endif
		}
	};
} // namespace mlp

#endif