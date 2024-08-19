#include "feed_forward.h"
#include <Eigen/Dense>

Eigen::MatrixXd FeedForwardNetwork(const Eigen::MatrixXd& input, int d_ff, int d_model) {
    Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(d_model, d_ff);
    Eigen::MatrixXd W2 = Eigen::MatrixXd::Random(d_ff, d_model);
    Eigen::MatrixXd b1 = Eigen::MatrixXd::Zero(1, d_ff);
    Eigen::MatrixXd b2 = Eigen::MatrixXd::Zero(1, d_model);

    Eigen::MatrixXd hidden = (input * W1 + b1.replicate(input.rows(), 1)).array().max(0);
    return hidden * W2 + b2.replicate(input.rows(), 1);
}
