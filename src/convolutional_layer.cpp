#include "convolutional_layer.h"

namespace mlp{
	void ConvolutionalLayer::init_weight(){
		uniform_rand(W_.begin(), W_.end(), -1, 1);
		uniform_rand(b_.begin(), b_.end(), -1, 1);
	}

	void ConvolutionalLayer::forward(){
		forward_cpu();
	}

	void ConvolutionalLayer::forward_cpu(){
		std::fill(output_.begin(), output_.end(), 0);
		for (size_t out = 0; out < out_maps; out++){  /* for each output feature map */
			for (size_t in = 0; in < in_maps; in++){  /* for each input feature map */
				for (size_t h_ = 0; h_ < out_height; h_++){
					for (size_t w_ = 0; w_ < out_width; w_++){
						output_[getOutIndex(out, h_, w_)] +=
							conv(getInforKernel(in, h_, w_), getW_(in, out));
					}
				}
			}
			/* use activate function to get output */
			for (size_t h_ = 0; h_ < out_height; h_++){
				for (size_t w_ = 0; w_ < out_width; w_++){
					output_[getOutIndex(out, h_, w_)] =
						a_->f(output_[getOutIndex(out, h_, w_)] + b_[getb_(out, h_, w_)]);
				}
			}
		}
	}
	/*
	void ConvolutionalLayer::fault_forward(){
		std::fill(output_.begin(), output_.end(), 0);
		for (size_t out = 0; out < out_maps; out++){  
			for (size_t in = 0; in < in_maps; in++){ 
				for (size_t h_ = 0; h_ < out_height; h_++){
					for (size_t w_ = 0; w_ < out_width; w_++){
						output_[getOutIndex(out, h_, w_)] +=
							conv(getInforKernel(in, h_, w_), getW_(in, out));
					}
				}
			}
			for (size_t h_ = 0; h_ < out_height; h_++){
				for (size_t w_ = 0; w_ < out_width; w_++){
					output_[getOutIndex(out, h_, w_)] =
						a_->f(output_[getOutIndex(out, h_, w_)] + b_[getb_(out, h_, w_)]);
				}
			}
		}
	}
	*/

	void ConvolutionalLayer::back_prop(){
		/*update err terms of this layer.*/
		for (size_t out = 0; out < out_maps; out++){
			for (size_t in = 0; in < in_maps; in++){
				for (size_t w_ = 0; w_ < out_width; w_++){
					for (size_t h_ = 0; h_ < out_height; h_++){
						for (size_t y_ = 0; y_ < kernel_size; y_++){
							for (size_t x_ = 0; x_ < kernel_size; x_++){
								auto ff = in * in_width * in_height + (h_ + y_) *
									in_width + (x_ + w_);
								g_[ff] += /*next layer err terms*/
									this->next->g_[out * out_width *
									out_height + h_ * out_width + w_] *
									/*weight*/
									W_[in * out_maps * kernel_size * kernel_size +
									out * kernel_size * kernel_size +
									kernel_size * (kernel_size - y_ - 1) +
									(kernel_size - 1 - x_)] *
									/*df of input*/
									a_->df(input_[ff]);
							}
						}
					}
				}
			}
		}

		/*update weight*/
		for (size_t out = 0; out < out_maps; out++){
			for (size_t in = 0; in < in_maps; in++){
				for (size_t h_ = 0; h_ < out_height; h_++){
					for (size_t w_ = 0; w_ < out_width; w_++){
						auto tt = getb_(out, h_, w_);
						for (size_t y_ = 0; y_ < kernel_size; y_++){
							for (size_t x_ = 0; x_ < kernel_size; x_++){
								/*find update pixel*/
								auto target = in * out_maps * kernel_size * kernel_size +
									out * kernel_size * kernel_size +
									kernel_size * (kernel_size - y_ - 1) +
									(kernel_size - 1 - x_);
								/*cal delta*/
								auto delta =
									/*learning rate*/
									alpha_ *
									/*input*/
									input_[in * in_width * in_height + (h_ + y_) *
									in_width + (x_ + w_)] *
									/*next layer err terms*/
									this->next->g_[tt]
									/*weight momentum*/
									+ lambda_ * deltaW_[target];

								W_[target] += delta;
								/*update momentum*/
								deltaW_[target] = delta;
							}
						}
						b_[tt] += alpha_ * this->next->g_[tt];
					}
				}
			}
		}
	}

	void ConvolutionalLayer::fix_backprop(){
		/*update err terms of this layer.*/
		for (size_t out = 0; out < out_maps; out++){
			for (size_t in = 0; in < in_maps; in++){
				for (size_t w_ = 0; w_ < out_width; w_++){
					for (size_t h_ = 0; h_ < out_height; h_++){
						for (size_t y_ = 0; y_ < kernel_size; y_++){
							for (size_t x_ = 0; x_ < kernel_size; x_++){
								auto ff = in * in_width * in_height + (h_ + y_) *
									in_width + (x_ + w_);
								g_[ff] += /*next layer err terms*/
									this->next->g_[out * out_width *
									out_height + h_ * out_width + w_] *
									/*weight*/
									W_[in * out_maps * kernel_size * kernel_size +
									out * kernel_size * kernel_size +
									kernel_size * (kernel_size - y_ - 1) +
									(kernel_size - 1 - x_)] *
									/*df of input*/
									a_->df(input_[ff]);
							}
						}
					}
				}
			}
		}

		/*update weight*/
		for (size_t out = 0; out < out_maps; out++){
			for (size_t in = 0; in < in_maps; in++){
				for (size_t h_ = 0; h_ < out_height; h_++){
					for (size_t w_ = 0; w_ < out_width; w_++){
						auto tt = getb_(out, h_, w_);
						for (size_t y_ = 0; y_ < kernel_size; y_++){
							for (size_t x_ = 0; x_ < kernel_size; x_++){
								/*find update pixel*/
								auto target = in * out_maps * kernel_size * kernel_size +
									out * kernel_size * kernel_size +
									kernel_size * (kernel_size - y_ - 1) +
									(kernel_size - 1 - x_);

								if (W_fix[target] == 1)continue;

								/*cal delta*/
								auto delta =
									/*learning rate*/
									alpha_ *
									/*input*/
									input_[in * in_width * in_height + (h_ + y_) *
									in_width + (x_ + w_)] *
									/*next layer err terms*/
									this->next->g_[tt]
									/*weight momentum*/
									+ lambda_ * deltaW_[target];

								W_[target] += delta;
								/*update momentum*/
								deltaW_[target] = delta;
							}
						}
						if (b_fix[tt] == 0) b_[tt] += alpha_ * this->next->g_[tt];
					}
				}
			}
		}
	}


}