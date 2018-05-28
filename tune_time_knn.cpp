#include "time_knn.hpp"
#include "serialize.hpp"
#include "output.hpp"
#include <stdlib.h>

int main() {
  float alpha[2] = {300, 400};
  int num_alpha = 2;
  float n_size[2] = {30, 35};
  int num_n_size = 2;
  float beta[3] = {750, 800, 850};
  int num_beta = 3;
  float gamma[3] = {-9, -8.5, -8};
  int num_gamma = 3;
  float delta[4] = {12, 13, 14, 15};
  int num_delta = 4;
  // Fit a model for each alpha value
  fprintf(stdout, "Alpha, n_size, tau, gamma, delta\n");
  for (int i = 0; i < num_alpha; i++) {
    // Unserialize it with each run since time_knn needs to store um_train
    struct dataset* um_train = unserialize("data/um_train.ser");
    struct dataset* mu_train = unserialize("data/mu_train.ser");
    struct dataset* probe = unserialize("data/um_probe.ser");
    TIME_KNN* time_knn = new TIME_KNN(n_size[0], alpha[i], 1, -1, beta[0], delta[0], gamma[0]);
    time_knn->fit(um_train, mu_train);
    // Now start adjusting parameters
    for (int n_size_index = 0; n_size_index < num_n_size; n_size_index++) {
      for (int beta_index = 0; beta_index < num_beta; beta_index++) {
        for (int gamma_index = 0; gamma_index < num_gamma; gamma_index++) {
          for (int delta_index = 0; delta_index < num_delta; delta_index++) {
            time_knn->set_n_size(n_size[n_size_index]);
            time_knn->set_tau(beta[beta_index]);
            time_knn->set_gamma(gamma[gamma_index]);
            time_knn->set_delta(delta[delta_index]);
            float score = time_knn->score(probe);
            fprintf(stdout,
            "%f, %f, %f, %f, %f, %f \n",
            alpha[i], n_size[n_size_index], beta[beta_index], gamma[gamma_index], delta[delta_index], score);
          }
        }
      }
    }
    delete um_train->data;
    delete um_train;
    delete mu_train->data;
    delete mu_train;
    delete probe->data;
    delete probe;
  }
}
