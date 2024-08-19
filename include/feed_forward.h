#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <Eigen/Dense>

Eigen::MatrixXd FeedForwardNetwork(const Eigen::MatrixXd& input, int d_ff, int d_model);

#endif 
