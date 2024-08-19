
INCLUDE_DIR="./include"
SRC_DIR="./src"
EXECUTABLE="transformer"

SRC_FILES="$SRC_DIR/attention.cpp $SRC_DIR/feed_forward.cpp $SRC_DIR/positional_encoding.cpp $SRC_DIR/transformer_encoder.cpp $SRC_DIR/main.cpp"

# Check if Eigen library is accessible (you can modify the path if necessary)
EIGEN_PATH="/usr/include/eigen3"

# Compilation command
g++ -I$INCLUDE_DIR -I$EIGEN_PATH $SRC_FILES -o $EXECUTABLE -std=c++11


if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    ./$EXECUTABLE
else
    echo "Compilation failed. Please check the code for errors."
fi
