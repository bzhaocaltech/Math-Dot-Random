#include "matrix.hpp"
#include <algorithm>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

/* Constructor class for matrix */
template <class T>
Matrix<T>::Matrix(int num_rows, int num_cols) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->matrix = new T[num_rows * num_cols];
    this->row_locks = new std::mutex*[num_rows];
    for (int i = 0; i < num_rows; i++) {
      this->row_locks[i] = new std::mutex();
    }
}

/* Construct a matrix from a file */
template <class T>
Matrix<T>::Matrix(string file) {
  ifstream ifs(file);
  boost::archive::binary_iarchive ia(ifs);

  /* Unserialize matrix parameters */
  int num_rows, num_cols;
  ia & num_rows;
  ia & num_cols;
  this->num_rows = num_rows;
  this->num_cols = num_cols;

  /* Unserialize data values */
  this->matrix = new T[num_rows * num_cols];
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      int new_num;
      ia & new_num;
      this->set_val(i, j, new_num);
    }
  }

  /* Create row locks */
  this->row_locks = new std::mutex*[num_rows];
  for (int i = 0; i < num_rows; i++) {
    this->row_locks[i] = new std::mutex();
  }
};

/* Serializes a matrix to a file */
template <class T>
void Matrix<T>::serialize(string file) {
  ofstream ofs(file);
  boost::archive::binary_oarchive oa(ofs);

  oa & this->num_rows;
  oa & this->num_cols;
  for (int i = 0; i < this->num_rows; i++) {
    for (int j = 0; j < this->num_cols; j++) {
      oa & this->get_val(i, j);
    }
  }
};

/* Returns a pointer to the beginning of a particular row of the matrix */
template <class T>
T* Matrix<T>::row(int row) {
    return this->matrix + (row * this->num_cols);
}

/* Returns the value of a particular element of the matrix */
template <class T>
T Matrix<T>::get_val(int row, int col) {
    return this->matrix[row * this->num_cols + col];
}

/* Sets the value of a particular element of the matrix */
template <class T>
void Matrix<T>::set_val(int row, int col, T val) {
    this->row_locks[row]->lock();
    this->matrix[row * this->num_cols + col] = val;
    this->row_locks[row]->unlock();
}

/* Returns number of rows */
template <class T>
int Matrix<T>::get_num_rows(void) {
  return num_rows;
}

/* Returns number of cols */
template <class T>
int Matrix<T>::get_num_cols(void) {
  return num_cols;
}

/* Multiplies this matrix by a scalar */
template <class T>
void Matrix<T>::mul_scalar(float scalar) {
    for (int i = 0; i < this->num_cols * this->num_rows; i++) {
        this->matrix[i] *= scalar;
    }
}

/* Destructor for matrix */
template <class T>
Matrix<T>::~Matrix() {
  for (int i = 0; i < num_rows; i++) {
    delete this->row_locks[i];
  }
  delete this->row_locks;
  delete this->matrix;
}

/* Updates with a matrix row pointed to by new_row. Frees new_row afterwards. */
template <class T>
void Matrix<T>::update_row(int row, T* new_row) {
    this->row_locks[row]->lock();
    T* matrix_row = this->row(row);
    for (int i = 0; i < this->num_cols; i++) {
        matrix_row[i] = new_row[i];
    }
    this->row_locks[row]->unlock();
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

/* Adds vec2 to vec1. */
float* vec_add(float* vec1, float* vec2, int length) {
    float* new_vec = new float[length];
    for (int i = 0; i < length; i++) {
        new_vec[i] = vec1[i] + vec2[i];
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

template class Matrix<int>;
template class Matrix<float>;
