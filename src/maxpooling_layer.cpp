#include "maxpooling_layer.h"

namespace mlp{

	void MaxpoolingLayer::forward(){
		forward_cpu();
	}

	void MaxpoolingLayer::forward_cpu(){
		for (size_t out = 0; out < out_maps; out++){
			for (size_t h_ = 0; h_ < in_height; h_ += 2){
				for (size_t w_ = 0; w_ < in_width; w_ += 2){
					output_[getOutIndex(out, h_, w_)] = max_In_(out, h_, w_,
						getOutIndex(out, h_, w_));
				}
			}
		}
	}

	void MaxpoolingLayer::back_prop(){
		for (auto pair : max_loc)
			g_[pair.second] = this->next->g_[pair.first];
	}
}