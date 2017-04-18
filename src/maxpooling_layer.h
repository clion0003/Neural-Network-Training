#ifndef MAXPOOLINGLAYER_H_
#define MAXPOOLINGLAYER_H_


#include <unordered_map>

#include "util.h"
#include "layer.h"

#pragma once
namespace mlp{
	class MaxpoolingLayer :public Layer
	{
	public:
		MaxpoolingLayer(size_t in_w, size_t in_h, size_t in_m, activation* a) :
			Layer(in_w * in_h * in_m, (in_w / 2) * (in_h / 2) * in_m, a)
		{
			in_width = in_w;
			in_height = in_h;
			in_maps = in_m;
			out_width = in_w / 2;
			out_height = in_h/2;
			out_maps = in_m;
			output_.resize(out_maps * out_width * out_height);
			g_.resize(in_width * in_height * in_maps);
		}

		void forward();

		void back_prop();

		void forward_cpu();

		void fix_backprop(){};

		void init_weight(){};

		void fault_forward(){};

		void generateFault_varition(float s){};

		void generateFault_sa(){};

		void find_fixed(int number){};

		void remap_best(){};

	private:
		int in_width, in_height, in_maps, out_width, out_height, out_maps;
		inline float_t max_In_(size_t in_index, size_t h_, size_t w_, size_t out_index){
			float_t max_pixel = 0;
			size_t tmp;
			for (size_t x = 0; x < 2; x++){
				for (size_t y = 0; y < 2; y++){
					tmp = (in_index * in_width * in_height) +
						((h_ + y) * in_width) + (w_ + x);
					if (max_pixel < input_[tmp]){
						max_pixel = input_[tmp];
						max_loc[out_index] = tmp;
					}
				}
			}
			return max_pixel;
		}


		inline size_t getOutIndex(size_t out, size_t h_, size_t w_){
			return out * out_width * out_height +
				h_ / 2 * out_width + (w_ / 2);
		}

		inline size_t getOutIndex_batch(size_t batch, size_t out, size_t h_, size_t w_){
			return batch* out_maps *out_width* out_height + out * out_width * out_height +
				h_ / 2 * out_width + (w_ / 2);
		}

		/*
		for each output, I store the connection index of the input,
		which will be used in the back propagation,
		for err translating.
		*/
		std::unordered_map<size_t, size_t> max_loc;
	};
}

#endif