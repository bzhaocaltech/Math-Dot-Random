/* Runs a simple blend. */

#include "simpleblend.hpp"
#include "serialize.hpp"
#include "output.hpp"

int main() {

  // the data to "train" on is a set of (num_predictions) x (num_models),
  // the list of predictions for each model, in a Matrix
  //
  struct dataset* x_train = unserialize("data/mu_train.ser");
  struct dataset* x_valid = unserialize("data/mu_valid.ser");

  // also obtain a vector of vector of floats, which is a vector containing
  // every model's vector of predictions
  vector<vector<float>*>* all_pred; // how to obtain?

  // the RSME of submitting an all-zeros solution to the scoreboard
  float zero_rsme; // have yet to set the value

  // arguments are num_models and num_predictions
  SimpleBlend* sb = new SimpleBlend(all_pred->at(0)->size(), all_pred->size());

  printf("Converting all predictions of all models into a matrix...\n");
  Matrix mat_pred = sb->prep_input(all_pred);
  printf("Done.\n")

  printf("Fitting the blend...\n");
  sb->fit(mat_pred, zero_rsme);
  printf("Done.\n");
  
  float score = sb->score(x_valid, mat_pred);
  printf("Out of sample MSE is %f\n", score);
  score = sb->score(x_train, mat_pred);
  printf("In sample MSE is %f\n", score);

  delete x_train->data;
  delete x_train;
  delete x_valid->data;
  delete x_valid;

  printf("Blending predictions for qual...\n");
  struct dataset* x_test = unserialize("data/mu_qual.ser");
 
  std::vector<float>* predictions = sb->predict(x_test, mat_pred);
  printf("Done.\n");

  output(*predictions, "results.dta");

  delete x_test->data;
  delete x_test;
  delete predictions;

  printf(":)\n");

  return 0;
}
