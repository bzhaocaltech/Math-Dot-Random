#include "better_blend.hpp"
#include "input.hpp"
#include "output.hpp"
#include "serialize.hpp"
#include <stdlib.h>


// NOTE: MAKE SURE EVERYTHING IS UM NOT MU
int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("USAGE: ./run_blend eta user_reg movie_reg epochs\n");
    exit(1);
  }
  // Get command line arguments
  float eta = atof(argv[1]);
  float user_reg = atof(argv[2]);
  float movie_reg = atof(argv[3]);
  int epochs = atoi(argv[4]);

  // Input this manually because i'm too lazy to do through command line
  int num_models = 3;
  char** probe_models = new char*[num_models];
  probe_models[0] = "results/knn_blend.dta";
  probe_models[1] = "results/svd_20_blend.dta";
  probe_models[2] = "results/mean_model_blend.dta";
  vector<float>** probe_data = input(num_models, probe_models);
  char** qual_models = new char*[num_models];
  qual_models[0] = "results/knn_results.dta";
  qual_models[1] = "results/svd_20_results.dta";
  qual_models[2] = "results/mean_model_results.dta";
  vector<float>** qual_data = input(num_models, qual_models);
  // Input results from python blend here. Or just pick values you think you like
  float* avg_constants = new float[num_models];
  avg_constants[0] = 0.45; // for knn
  avg_constants[1] = 0.53; // for svd
  avg_constants[2] = 0.02; // for mean model

  struct dataset* x_probe = unserialize("data/mu_probe.ser");
  struct dataset* x_test = unserialize("data/mu_qual.ser");

  better_blend* bb = new better_blend(eta, user_reg, movie_reg, num_models,
  avg_constants, probe_data, qual_data);
  bb->fit(x_probe, epochs);
  vector<float>* predictions = bb->predict(x_test);
  output(*predictions, "blended_results.dta");

  delete x_probe->data;
  delete x_probe;
  delete x_test->data;
  delete x_test;
  delete predictions;

  // svd->serialize(file_name);

  return 0;
}
