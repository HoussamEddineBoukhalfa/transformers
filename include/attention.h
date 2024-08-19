#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>

Eigen::MatrixXd ScaledDotProductAttention(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V);
Eigen::MatrixXd MultiHeadAttention(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, int num_heads);

#endif 