/* Runs a KNN */
/* TODO: Serialize the model */

#include "knn.hpp"
#include "serialize.hpp"
#include "output.hpp"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("USAGE: ./run_knn neighborhood_size alpha e\n");
    exit(1);
  }
  // Get command line arguments
  int n_size = atoi(argv[1]);
  float alpha = atof(argv[2]);
  float e = atof(argv[3]);

  struct dataset* um_train = unserialize("data/um_train.ser");
  struct dataset* mu_train = unserialize("data/mu_train.ser");
  struct dataset* valid = unserialize("data/um_valid.ser");
  struct dataset* probe = unserialize("data/um_probe.ser");

  KNN* knn = new KNN(n_size, alpha, e);
  knn->fit(um_train, mu_train);
  // float score = knn->score(um_train);
  // printf("In sample RMSE is %f\n", score);
  float score = knn->score(valid);
  printf("Out of sample RMSE is %f\n", score);
  score = knn->score(probe);
  printf("Probe RMSE is %f\n", score);

  struct dataset* test = unserialize("data/um_qual.ser");
  std::vector<float>* predictions = knn->predict(test);

  output(*predictions, "results.dta");

  delete test->data;
  delete test;
  delete mu_train->data;
  delete mu_train;
  delete um_train->data;
  delete um_train;
  delete valid->data;
  delete valid;
  delete probe->data;
  delete probe;
  delete predictions;

  return 0;
}
