/* Simple linear regression blending of model outputs. */

#include "model.hpp"
#include "matrix.hpp"


class SimpleBlend: public Blend {
  private:
    /* The vector containing the weight assigned to each model. */
    float* weights;

    // Number of models
    int num_models;

    // Number of predictions
    int num_predictions;

  public:
    /* Constructor for the simple blend */
    SimpleBlend(int num_predictions, int num_models);

    /* Combine model predictions into one matrix. */
    Matrix prep_input(vector<vector<float>*>);

    /* Fits the weights of the blend. */
    void fit(Matrix A, float zero_sol);

    /* Predicts on the dataset using each model's prediction and the 
     * blend-model weights */
    vector<float>* predict(Matrix A);

    /* Destructor for simple blend */
    ~SimpleBlend();
};
