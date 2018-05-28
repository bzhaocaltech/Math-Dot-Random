/* Runs a Time SVDPP */

#include "time_svdpp.hpp"
#include "serialize.hpp"
#include "output.hpp"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("USAGE: ./run_time_svdpp #latent_factors eta regularization #epochs\n");
    exit(1);
  }
  // Get command line arguments
  int latent_factors = atoi(argv[1]);
  float eta = atof(argv[2]);
  float reg = atof(argv[3]);
  int epochs = atoi(argv[4]);
  // string file_name = string("models/svd_") + argv[1] + "_" + argv[2] + "_" + argv[3]
  // + "_" + argv[4] + ".ser";

  struct dataset* x_train = unserialize("data/um_train.ser");
  struct dataset* x_valid = unserialize("data/um_valid.ser");
  struct dataset* x_probe = unserialize("data/um_probe.ser");

  TIME_SVDPP* time_svdpp = new TIME_SVDPP(latent_factors, eta, reg);
  time_svdpp->fit(x_train, epochs, x_probe);
  float score = time_svdpp->score(x_train);
  printf("In sample RMSE is %f\n", score);
  score = time_svdpp->score(x_valid);
  printf("Out of sample RMSE is %f\n", score);
  score = time_svdpp->score(x_probe);
  printf("Probe RMSE is %f\n", score);

  delete x_train->data;
  delete x_train;
  delete x_valid->data;
  delete x_valid;

  struct dataset* x_test = unserialize("data/um_qual.ser");
  std::vector<float>* predictions = time_svdpp->predict(x_test);

  std::vector<float>* probe_predictions = time_svdpp->predict(x_probe);

  output(*predictions, "results.dta");
  output(*probe_predictions, "probe.dta");

  delete x_test->data;
  delete x_test;
  delete x_probe->data;
  delete x_probe;
  delete predictions;

  return 0;
}
