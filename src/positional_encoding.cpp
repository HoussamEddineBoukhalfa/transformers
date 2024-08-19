#include "positional_encoding.h"
#include <cmath>

Eigen::MatrixXd PositionalEncoding(int sequence_length, int d_model) {
    Eigen::MatrixXd pos_enc(sequence_length, d_model);
    for (int pos = 0; pos < sequence_length; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            if (i % 2 == 0) {
                pos_enc(pos, i) = sin(pos / pow(10000, 2.0 * i / d_model));
            } else {
                pos_enc(pos, i) = cos(pos / pow(10000, 2.0 * (i - 1) / d_model));
            }
        }
    }
    return pos_enc;
}
