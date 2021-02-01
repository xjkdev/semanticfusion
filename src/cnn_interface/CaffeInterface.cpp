/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is SemanticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "CaffeInterface.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <fmt/format.h>
#include <torch/script.h>


bool CaffeInterface::Init(const std::string& model_path, const std::string& weights) {
  network_.reset(new torch::jit::script::Module);
  *network_ = torch::jit::load(model_path);
  // torch::load(*network_, model_path);
  network_->to(device_);
  return true;
}

int CaffeInterface::num_output_classes() {
  return 14;
}

std::shared_ptr<torch::Tensor> CaffeInterface::ProcessFrame(const ImagePtr rgb, const DepthPtr depth,
                                  const int height, const int width) {
  if (!initialised_) {
    return std::shared_ptr<torch::Tensor>();
  }
  cv::Mat input_image(height,width,CV_8UC3, rgb);
  cv::Mat input_depth(height,width,CV_16UC1, depth);

  torch::Tensor inputs = torch::zeros({4,224,224}, torch::ScalarType::Float);

  cv::Mat resized_image(network_height,network_width,CV_8UC3);
  cv::resize(input_image,resized_image,resized_image.size(),0,0);
  cv::Mat resized_depth(network_height,network_width,CV_16UC1);
  cv::resize(input_depth,resized_depth,resized_depth.size(),0,0,cv::INTER_NEAREST);

  // This performs inpainting of the depth map on the fly, however for NYU
  // experiments we used the same matlab inpainting given with the toolkit, so
  // the input to the CNN is the same as with the normal NYU baseline results
  cv::Mat depthf(network_width,network_height, CV_8UC1);
  resized_depth.convertTo(depthf, CV_8UC1, 25.0/1000.0);
  const unsigned char noDepth = 0;
  cv::inpaint(depthf, (depthf == noDepth), depthf, 5.0, cv::INPAINT_TELEA);

  float* input_data = inputs.data_ptr<float>();
  const float mean[] = {104.0,117.0,123.0};

  for (int h = 0; h < network_height; ++h) {
    const uchar* image_ptr = resized_image.ptr<uchar>(h);
    const uint16_t* depth_ptr = resized_depth.ptr<uint16_t>(h);
    int image_index = 0;
    int depth_index = 0;
    for (int w = 0; w < network_width; ++w) {
      float r = static_cast<float>(image_ptr[image_index++]);
      float g = static_cast<float>(image_ptr[image_index++]);
      float b = static_cast<float>(image_ptr[image_index++]);
      const int b_offset = ((0 * network_height) + h) * network_width + w;
      const int g_offset = ((1 * network_height) + h) * network_width + w;
      const int r_offset = ((2 * network_height) + h) * network_width + w;
      input_data[b_offset] = b - mean[0];
      input_data[g_offset] = g - mean[1];
      input_data[r_offset] = r - mean[2];
      // if (inputs->channels() == 4) {
        //Convert to m from mm, and remove mean of 2.8m
      float d = static_cast<float>(depth_ptr[depth_index++]) * (1.0f/1000.0f) ;
      const int d_offset = ((3 * network_height) + h) * network_width + w;
      input_data[d_offset] = d - 2.841;
      // }
    }
  }
  std::vector<torch::jit::IValue> net_inputs;
  inputs.to(device_);
  net_inputs.push_back(inputs);
  const torch::Tensor output = network_->forward(net_inputs).toTensor().unsqueeze(0);
  fmt::print("{}\n", fmt::join(output.sizes().vec(),","));
  if (!output_probabilities_) {
    output_probabilities_.reset(new torch::Tensor);
  }
  *output_probabilities_ = output.unsqueeze(0);
  return output_probabilities_;
}
