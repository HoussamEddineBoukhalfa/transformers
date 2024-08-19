#ifndef TRANSFORMER_ENCODER_H
#define TRANSFORMER_ENCODER_H

#include <Eigen/Dense>

Eigen::MatrixXd TransformerEncoderLayer(const Eigen::MatrixXd& input, int num_heads, int d_ff);

#endif
