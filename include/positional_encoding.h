#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <Eigen/Dense>

Eigen::MatrixXd PositionalEncoding(int sequence_length, int d_model);

#endif 
