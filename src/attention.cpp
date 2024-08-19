#include "attention.h"
#include <cmath>

Eigen::MatrixXd ScaledDotProductAttention(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V) {
    double d_k = K.cols();
    Eigen::MatrixXd scores = (Q * K.transpose()) / sqrt(d_k);
    Eigen::MatrixXd softmaxScores = scores.array().exp().matrix();
    Eigen::ArrayXd rowSum = softmaxScores.rowwise().sum();
    softmaxScores = (softmaxScores.array().colwise() / rowSum).matrix();
    return softmaxScores * V;
}


Eigen::MatrixXd MultiHeadAttention(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, int num_heads) {
    int d_model = Q.cols();
    int d_k = d_model / num_heads;
    Eigen::MatrixXd output(Q.rows(), d_model);

    for (int i = 0; i < num_heads; ++i) {
        Eigen::MatrixXd Q_head = Q.middleCols(i * d_k, d_k);
        Eigen::MatrixXd K_head = K.middleCols(i * d_k, d_k);
        Eigen::MatrixXd V_head = V.middleCols(i * d_k, d_k);

        output.middleCols(i * d_k, d_k) = ScaledDotProductAttention(Q_head, K_head, V_head);
    }

    return output;
}
