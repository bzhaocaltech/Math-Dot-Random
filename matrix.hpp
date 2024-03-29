/* Simple class for matrix operations. Also contains a lot of vector operations */
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <string>

#include <vector>
#include <mutex>

using namespace std;

template <class T> class Matrix {
  private:
    int num_rows;
    int num_cols;
    T* matrix;
    std::mutex** row_locks;
  public:
    /* Constructor class for matrix */
    Matrix(int num_rows, int num_cols);

    /* Construct a matrix from a file */
    Matrix(string file);

    /* Returns a pointer to the beginning of a particular row of the matrix */
    T* row(int row);

    /* Returns the value of a particular element of the matrix */
    T get_val(int row, int col);

    /* Sets the value of a particular element of the matrix */
    void set_val(int row, int col, T val);

    /* Returns number of rows */
    int get_num_rows(void);

    /* Returns number of cols */
    int get_num_cols(void);

    /* Multiplies this matrix by a scalar */
    void mul_scalar(float scalar);

    /* Updates with a matrix row pointed to by new_row. Frees new_row afterwards. */
    void update_row(int row, T* new_row);

    /* Serializes a matrix to a file */
    void serialize(string file);

    /* Destructor for matrix */
    ~Matrix();
};

/* Takes the dot product of two vectors */
float dot_prod(float* vec1, float* vec2, int length);

/* Subtracts vec2 from vec1. */
float* vec_sub(float* vec1, float* vec2, int length);

/* Adds vec2 to vec1. */
float* vec_add(float* vec1, float* vec2, int length);

/* Multiplies a vector by a scalar. Returns a new float* */
float* scalar_vec_prod(float scalar, float* vec1, int length);

#endif
