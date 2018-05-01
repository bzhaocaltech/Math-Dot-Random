/* Simple class for matrix operations. Also contains a lot of vector operations */

using namespace std;

class Matrix {
  private:
    int num_rows;
    int num_cols;
    float* matrix;
  public:
    /* Constructor class for matrix */
    Matrix(int num_rows, int num_cols);

    /* Returns a pointer to the beginning of a particular row of the matrix */
    float* row(int row);

    /* Returns the value of a particular element of the matrix */
    int get_val(int row, int col);

    /* Sets the value of a particular element of the matrix */
    void set_val(int row, int col, float val);

    /* Multiplies this matrix by a scalar */
    void mul_scalar(float scalar);

    /* Updates with a matrix row pointed to by new_row. Frees new_row afterwards. */
    void update_row(int row, float* new_row);

    /* Destructor for matrix */
    ~Matrix();
};

/* Takes the dot product of two vectors */
float dot_prod(float* vec1, float* vec2, int length);

/* Subtracts vec2 from vec1. */
float* vec_sub(float* vec1, float* vec2, int length);

/* Multiplies a vector by a scalar. Returns a new float* */
float* scalar_vec_prod(float scalar, float* vec1, int length);