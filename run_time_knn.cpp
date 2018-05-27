/* Runs a KNN */
/* TODO: Serialize the model */

#include "time_knn.hpp"
#include "serialize.hpp"
#include "output.hpp"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("USAGE: ./run_knn neighborhood_size alpha tau delta gamma\n");
    exit(1);
  }
  // Get command line arguments
  int n_size = atoi(argv[1]);
  int alpha = atoi(argv[2]);
  float e = 1;
  float min_pearson = -1;
  float tau = atof(argv[3]);
  float delta = atof(argv[4]);
  float gamma = atof(argv[5]);

  struct dataset* um_train = unserialize("data/um_train.ser");
  struct dataset* mu_train = unserialize("data/mu_train.ser");
  struct dataset* probe = unserialize("data/um_probe.ser");

  TIME_KNN* time_knn = new TIME_KNN(n_size, alpha, e, min_pearson, tau, delta, gamma);
  time_knn->fit(um_train, mu_train);
  float score = time_knn->score(probe);
  printf("Probe RMSE is %f\n", score);

  struct dataset* test = unserialize("data/um_qual.ser");
  std::vector<float>* predictions = time_knn->predict(test);

  output(*predictions, "results.dta");

  delete test->data;
  delete test;
  delete mu_train->data;
  delete mu_train;
  delete um_train->data;
  delete um_train;
  delete probe->data;
  delete probe;
  delete predictions;

  return 0;
}
