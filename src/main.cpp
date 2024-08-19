#include <iostream>
#include <Eigen/Dense>
#include "attention.h"
#include "feed_forward.h"
#include "positional_encoding.h"
#include "transformer_encoder.h"

int main() {
    int sequence_length = 10;  // Example sequence length
    int d_model = 512;         // Example model dimension
    int num_heads = 8;         // Example number of attention heads
    int d_ff = 2048;           // Example feed-forward dimension

    // Example input (randomly initialized)
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(sequence_length, d_model);

    // Positional encoding
    Eigen::MatrixXd pos_enc = PositionalEncoding(sequence_length, d_model);

    // Add positional encoding to input
    input += pos_enc;

    // Transformer Encoder Layer
    Eigen::MatrixXd output = TransformerEncoderLayer(input, num_heads, d_ff);

    // Print the output
    std::cout << "Transformer Encoder Output:\n" << output << std::endl;

    return 0;
}
