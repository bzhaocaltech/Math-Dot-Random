#include "matrix.hpp"

/* Constructor class for matrix */
Matrix::Matrix(int num_rows, int num_cols) {
  this->num_rows = num_rows;
  this->num_cols = num_cols;
  this->matrix = new float[num_rows * num_cols];
}

/* Construct a matrix from a boost binary_iarchive */
Matrix::Matrix(boost::archive::binary_iarchive ia) {
  /* Unserialize matrix parameters */
  int num_rows, num_cols;
  ia & num_rows;
  ia & num_cols;
  this->num_rows = num_rows;
  this->num_cols = num_cols;

  /* Unserialize data values */
  this->matrix = new float[num_rows * num_cols];
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      int new_num;
      ia & new_num;
      this->set_val(i, j, new_num);
    }
  }
};

/* Serializes a matrix to a given file. */
void Matrix::serialize(boost::archive::binary_oarchive oa) {
  oa & this->num_rows;
  oa & this->num_cols;
  for (int i = 0; i < this->num_rows; i++) {
    for (int j = 0; j < this->num_cols; j++) {
      oa & this->get_val(i, j);
    }
  }
};

/* Returns a pointer to the beginning of a particular row of the matrix */
float* Matrix::row(int row) {
  return this->matrix + (row * this->num_cols);
}

/* Returns the value of a particular element of the matrix */
int Matrix::get_val(int row, int col) {
  return this->matrix[row * this->num_cols + col];
}

/* Sets the value of a particular element of the matrix */
void Matrix::set_val(int row, int col, float val) {
  this->matrix[row * this->num_cols + col] = val;
}

/* Returns number of rows */
int Matrix::get_num_rows(void) {
  return num_rows;
}

/* Returns number of cols */
int Matrix::get_num_cols(void) {
  return num_cols;
}

/* Multiplies this matrix by a scalar */
void Matrix::mul_scalar(float scalar) {
  for (int i = 0; i < this->num_cols * this->num_rows; i++) {
    this->matrix[i] *= scalar;
  }
}

/* Destructor for matrix */
Matrix::~Matrix() {
  delete this->matrix;
}

/* Updates with a matrix row pointed to by new_row. Frees new_row afterwards. */
void Matrix::update_row(int row, float* new_row) {
  float* matrix_row = this->row(row);
  for (int i = 0; i < this->num_cols; i++) {
    matrix_row[i] = new_row[i];
  }
  delete new_row;
};

/* Takes the dot product of two vectors */
float dot_prod(float* vec1, float* vec2, int length) {
  float sum = 0;
  for (int i = 0; i < length; i++) {
    sum += vec1[i] * vec2[i];
  }
  return sum;
}

/* Subtracts vec2 from vec1. */
float* vec_sub(float* vec1, float* vec2, int length) {
  float* new_vec = new float[length];
  for (int i = 0; i < length; i++) {
    new_vec[i] = vec1[i] - vec2[i];
  }
  return new_vec;
}

/* Multiplies a vector by a scalar. Returns a new float* */
float* scalar_vec_prod(float scalar, float* vec1, int length) {
  float* vec2 = new float[length];
  for (int i = 0; i < length; i++) {
    vec2[i] = vec1[i] * scalar;
  }
  return vec2;
}
