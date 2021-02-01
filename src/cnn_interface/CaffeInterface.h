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

#ifndef CAFFE_INTERFACE_H_
#define CAFFE_INTERFACE_H_
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <torch/torch.h>
#include <utilities/Types.h>
#include <utilities/Array.h>

class CaffeInterface {
public:
  CaffeInterface() : initialised_(false) {}
  virtual ~CaffeInterface() {}
  bool Init(const std::string& model_path, const std::string& weights);
  std::shared_ptr<torch::Tensor> ProcessFrame(
                            const ImagePtr rgb, const DepthPtr depth,
                            const int height, const int width);
  int num_output_classes();
private:
  bool initialised_;
  // std::unique_ptr<DeConvNet> network_;
  const torch::Device device_ = c10::kCUDA;
  std::unique_ptr<torch::jit::script::Module> network_;
  std::shared_ptr<torch::Tensor> output_probabilities_;

  const int network_width = 224;
  const int network_height = 224;
};

#endif /* CAFFE_INTERFACE_H_ */
