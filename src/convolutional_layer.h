#ifndef CONVOLUTIONALLAYER_H_
#define CONVOLUTIONALLAYER_H_

#pragma once
#include "layer.h"
#include "util.h"
#include <math.h>
#include <vector>

#ifdef GPU
#include "cuda_runtime.h"
#include "cublas_v2.h"
#endif


namespace mlp{

	class ConvolutionalLayer :public Layer
	{
	public:
		ConvolutionalLayer(size_t in_w, size_t in_h, size_t in_m,
			size_t kernel_s, size_t out_m, activation* a) :
			Layer(in_w * in_h * in_m, (in_w - kernel_s + 1) * (in_h - kernel_s + 1) * out_m, a)
		{
			in_width = in_w;
			in_height = in_h;
			in_maps = in_m;
			out_width = in_w - kernel_s + 1;
			out_height = in_h - kernel_s + 1;
			out_maps = out_m;
			kernel_size = kernel_s;

			W_.resize(kernel_size * kernel_size * in_maps * out_maps);
			W_fix.resize(kernel_size * kernel_size * in_maps * out_maps);
			deltaW_.resize(kernel_size * kernel_size * in_maps * out_maps);
			b_.resize(out_maps * out_width * out_height);
			b_fix.resize(out_maps * out_width * out_height);
			output_.resize(out_maps * out_width * out_height);
			g_.resize(in_width * in_height * in_maps);
			this->init_weight();
		}

		void init_weight();

		void forward();

		//void fault_forward();

		void forward_cpu();

		void back_prop();

		void fix_backprop();

		void fault_forward(){};

		void generateFault_varition(float s){};

		void generateFault_sa(){};

		void find_fixed(int number){};

		void remap_best(){};


	private:
		int in_width, in_height, in_maps, out_width, out_height, out_maps;
		int kernel_size;


		inline size_t getOutIndex(size_t out, size_t h_, size_t w_){
			return out * out_height * out_width + h_ * out_width + w_;
		}

		inline vec_t getInforKernel(size_t in, size_t h_, size_t w_){
			vec_t r;
			for (size_t y = 0; y < kernel_size; y++){
				for (size_t x = 0; x < kernel_size; x++){
					r.push_back(input_[in * (in_width * in_height) + (h_ + y) * in_width + x + w_]);
				}
			}
			return r;
		}

		inline vec_t getW_(size_t in, size_t out){
			vec_t r;
			for (size_t i = 0; i < kernel_size * kernel_size; i++)
				r.push_back(W_[in * out_maps * kernel_size * kernel_size
				+ out * kernel_size * kernel_size + i]);
			return r;
		}

		inline int getb_(size_t out, size_t h_, size_t w_){
			return out * out_width * out_height + h_ * out_height + w_;
		}

		float_t conv(vec_t a, vec_t b){
			assert(a.size() == b.size());
			float_t sum = 0, size = a.size();
			for (size_t i = 0; i < size; i++){
				sum += a[i] * b[size - i - 1];
			}
			return sum;
		}

	};
}
#endif