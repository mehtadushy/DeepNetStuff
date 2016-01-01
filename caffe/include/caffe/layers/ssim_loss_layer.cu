#include <vector>

#include "caffe/layers/ssim_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void CudaGaussConvolveHelper(const int nthreads,
    const Dtype* const in_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, 
    const double* const gauss_kernel, Dtype* const out_data ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int wstart = pw * stride_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    Dtype aveval = 0;
    const Dtype* const in_slice =
        in_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += gauss_kernel[(h-hstart)*kernel_w_+(w-wstart)] * in_slice[h * width + w];
      }
    }
    out_data[index] = aveval;
  }
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::CudaGaussConvolveHelper(const Blob<Dtype>>& in,
    Blob<Dtype>& out) {	
  //Parallelized on the # of outputs to be produced
  CudaGaussConvolveHelper<Dtype><<<CAFFE_GET_BLOCKS(out.count(), CAFFE_CUDA_NUM_THREADS>>>(
        out.count(), in.gpu_data(), in.num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, gauss_kernel_.gpu_data(),
	out.mutable_gpu_data() 
	);
}
template <typename Dtype>
void SSIMLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {	
  int count = bottom[0]->count();
  CudaGaussConvolveHelper(*bottom[0],ux_);
  CudaGaussConvolveHelper(*bottom[0],uy_);

  Blob<Dtype> tempContainer1, tempContainer2;
  tempContainer1.ReshapeLike(*bottom[0]);
  caffe_gpu_sqr(count, bottom[0]->gpu_data(), tempContainer1.mutable_gpu_data()); 
  CudaGaussConvolveHelper(tempContainer1,sx2_);
  caffe_gpu_sqr(count, bottom[1]->gpu_data(), tempContainer1.mutable_gpu_data()); 
  CudaGaussConvolveHelper(tempContainer1,sy2_);
  caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), tempContainer1.mutable_gpu_data()); 
  CudaGaussConvolveHelper(tempContainer1,sxy_);

  tempContainer1.ReshapeLike(ux_);
  tempContainer2.ReshapeLike(uy_);
  count = tempContainer1.count();

  //Compute ux^2 and uy^2 and collect ux^2+uy^2 for later use
  caffe_gpu_sqr(count, ux_.gpu_data(), tempContainer1.mutable_gpu_data());
  caffe_gpu_sub(count, sx2_.gpu_data(), tempContainer1.gpu_data(), sx2_.mutable_gpu_data());
  caffe_gpu_sqr(count, uy_.gpu_data(), tempContainer2.mutable_gpu_data());
  caffe_gpu_sub(count, sy2_.gpu_data(), tempContainer2.gpu_data(), sy2_.mutable_gpu_data());
  caffe_gpu_add(count, tempContainer1.gpu_data(), tempContainer2.gpu_data(), tempContainer2.mutable_gpu_data());

  caffe_gpu_mul(count, ux_.gpu_data(), uy_.gpu_data(), tempContainer1.mutable_gpu_data());
  caffe_gpu_sub(count, sxy_.gpu_data(), tempContainer1.gpu_data(), sxy_.mutable_gpu_data());
  
  const Dtype C1 = c1_;
  caffe_gpu_scale(count, Dtype(2), tempContainer1.gpu_data(), tempContainer1.mutable_gpu_data());
  caffe_gpu_add_scalar(count, C1, tempContainer1.mutable_gpu_data());
  caffe_gpu_add_scalar(count, C1, tempContainer2.mutable_gpu_data());
  caffe_gpu_div(count, tempContainer1.gpu_data(), tempContainer2.gpu_data(), lp_.mutable_gpu_data());

  const Dtype C2 = c2_;
  caffe_gpu_add(count, sx2_.gpu_data(), sy2_.gpu_data(), tempContainer2.mutable_gpu_data()); 
  caffe_gpu_add_scalar(count, C2, tempContainer2.mutable_gpu_data());
  caffe_gpu_axpby(count, Dtype(2), sxy_.gpu_data(), Dtype(0), tempContainer1.mutable_gpu_data());
  caffe_gpu_add_scalar(count, C2, tempContainer1.mutable_gpu_data());
  caffe_gpu_div(count, tempContainer1.gpu_data(), tempContainer2.gpu_data(), cs_.mutable_gpu_data());
  
  Dtype ssim = caffe_gpu_dot(count, lp_.gpu_data(),cs_.gpu_data()) / bottom[0]->num();
  Dtype loss = Dtype(1)-ssim;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void SSIMBackward(const int nthreads, 
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const Dtype* const x,const Dtype* const y,
    const Dtype* const ux,const Dtype* const uy,
    const Dtype* const sx2, const Dtype* const sy2, const Dtype* const sxy,
    const Dtype* const lp, const Dtype* const cs, 
    const double* const gauss_kernel,  Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width ;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int q = h* width + w; 
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
	int p = ph * pooled_width_ + pw;
        int hstart = ph * stride_h ;
        int wstart = pw * stride_w ;
        int hend = min(hstart + kernel_h, height );
        int wend = min(wstart + kernel_w, width);
	Dtype deriv1 = Dtype(2) * cs[p] * (uy[p]-ux[p]*lp[p]) / (ux[p]*ux[p]+uy[p]*uy[p]+Dtype(c1_));
	Dtype deriv2 = Dtype(2) * lp[p]  / (sx2[p]+sy2[p]+Dtype(c2_));
        gradient += top_diff_slice[p] ;
      }
    }
    bottom_diff[index] = gradient;
  }
}
template <typename Dtype>
void SSIMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const Dtype alpha = -top[0]->cpu_diff()[0] / bottom[0]->num();

  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
         for (int pw = 0; pw < pooled_width_; ++pw) {
	    int hstart = ph * stride_h_ ;
	    int wstart = pw * stride_w_ ;
	    int hend = min(hstart + kernel_h_, height_);
	    int wend = min(wstart + kernel_w_, width_ );
	    int p = ph * pooled_width_ + pw;
	    Dtype deriv1 = Dtype(2) * cs[p] * (uy[p]-ux[p]*lp[p]) / (ux[p]*ux[p]+uy[p]*uy[p]+Dtype(c1_));
	    Dtype deriv2 = Dtype(2) * lp[p]  / (sx2[p]+sy2[p]+Dtype(c2_));
	    for (int h = hstart; h < hend; ++h) {
	      for (int w = wstart; w < wend; ++w) {
		int q = h * width_ + w;
		bottom_diff[q] += 
                    gaussian[(h-hstart)*kernel_w_+(w-wstart)]* ( deriv1 + ( deriv2 * ((y[q] - uy[p]) - cs[p]*(x[q]-ux[p]))));
	      }
	    }
	  }
	}
	// offset
	x+= bottom[0]->offset(0,1);
	y+= bottom[1]->offset(0,1);
	bottom_diff += bottom[0]->offset(0, 1);
	ux+= ux_.offset(0,1);
	uy+= ux_.offset(0,1);
	sx2+= ux_.offset(0,1);
	sy2+= ux_.offset(0,1);
	sxy+= ux_.offset(0,1);
	lp+= ux_.offset(0,1);
	cs+= ux_.offset(0,1);
    }
  }
  caffe_cpu_scale(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          bottom[0]->cpu_diff(),              // x
          bottom_diff);  // y
}

template <typename Dtype>
void SSIMLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_sign(bottom[i]->count(), diff_.gpu_data(),
      		     bottom[i]->mutable_gpu_diff());
      caffe_gpu_scale(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          bottom[i]->gpu_diff(),                   // x
          bottom[i]->mutable_gpu_diff());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SSIMLossLayer);

}  // namespace caffe
