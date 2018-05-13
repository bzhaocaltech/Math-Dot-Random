#include "simpleblend.hpp"
#include "matrix_helper.cpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

SimpleBlend::SimpleBlend(int num_predictions, int num_models) {
	this->weights = new float[num_models];
	this->num_models = num_models;
	this->num_predictions = num_predictions;
}

Matrix SimpleBlend::prep_input(vector<vector<float>*>* preds) {
	// unpacks a vector of vector of floats into a matrix where each row is
	// a set of predictions by all models for 1 data point and each column 
	// is a set of preditions on all data points made by 1 model

	Matrix mat = new Matrix(preds->at(0)->size(), preds->size());

	for (int i = 0; i < preds->size(); i++)
	{
		for (int i = 0; i < preds->at(0)->size(); ++i)
		{
			mat->set_val(i, j, preds->at(i)->preds->at(j))
		}
	}
	return mat;
}

/* Given a set of (num_predictions) x (num_models), get the blend weights. */
void SimpleBlend::fit(Matrix A, float zero_sol) {
	// zero_sol is the const RMSE result of submitting all 0 solution

	// TODO -- inverse function is probably not ready, is in 
	// matrix_helper.cpp, borrowed from elsewhere.
	// (A^T * A)^-1 
	Matrix first_half = inverse(A->mat_mul(A->get_transpose()));

	// A^T * s is #models x vector
	float second_half[num_models];

	for(int i = 0; i < this->num_models; k++) {

		// sum of all predictions squared, for i_th model 
		float pred_cum = 0.0;
		for (int j = 0; j < this->num_predictions; j++) {
			pred_cum += pow(A[j][i]);
		}

		second_half[i] = 0.5 * (pred_cum + zero_sol);
	}	

	// multiply matrix first-half by vector second-half
	// this gives us weights, so store 
	// ALSO need to extract the float vector from the MM result
	// because weights is a float vector
	for (int i = 0; i < num_models; i++)
	{
		this->weights[i] = dot_prod(first_half->row(i), second_half);
	}


}

vector<float>* SimpleBlend::predict(Matrix A) {
	vector<float>* predictions = new vector<float>();
    for (int i = 0; i < A->num_predictions; i++) {
    	float* point = A->row(i);
    	// we want to transpose the weight vector and mult with a row of
    	// A, which would be one prediction across all models
        float p = dot_prod(this->weights, point, this->num_models);
        predictions->push_back(p);
    }
    return predictions;
}

SimpleBlend::~SimpleBlend() {
	delete this->weights;
}