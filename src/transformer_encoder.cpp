#include "transformer_encoder.h"
#include "attention.h"
#include "feed_forward.h"

Eigen::MatrixXd TransformerEncoderLayer(const Eigen::MatrixXd& input, int num_heads, int d_ff) {
    // Multi-head attention
    Eigen::MatrixXd attention_output = MultiHeadAttention(input, input, input, num_heads);

    // Add & Norm (Residual Connection + Layer Normalization)
    Eigen::MatrixXd norm1 = attention_output + input;

    // Feed-forward network
    Eigen::MatrixXd ff_output = FeedForwardNetwork(norm1, d_ff, input.cols());

    // Add & Norm (Residual Connection + Layer Normalization)
    return ff_output + norm1;
}
