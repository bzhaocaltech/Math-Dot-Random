/* Runs a simple SVD */

#include "svd.hpp"
#include "serialize.hpp"
#include "output.hpp"

int main() {
  std::vector<int*>* x_train = unserialize("data/mu_train.ser", 4);
  std::vector<int*>* x_valid = unserialize("data/mu_valid.ser", 4);

  SVD* svd = new SVD(10, 0.03, 0.01);
  svd->fit(x_train, 10);
  float score = svd->score(x_valid);
  printf("Out of sample MSE is %f\n", score);
  score = svd->score(x_train);
  printf("In sample MSE is %f\n", score);

  std::vector<int*>* x_test = unserialize("data/mu_qual.ser", 3);
  std::vector<float>* predictions = svd->predict(x_test);

  output(*predictions, "results.dta");

  free(x_train);
  free(x_valid);
  free(x_test);
  free(predictions);

  return 0;
}
