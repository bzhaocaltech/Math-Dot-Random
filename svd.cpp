/* A basic matrix factorization model */

#include "svd.hpp"

using namespace boost::numeric::ublas;
SVD::SVD(int latent_factors, int eta, int num_of_movies = NUM_MOVIES,
        int num_of_users = NUM_USERS) {
    this->U = new matrix<int> (num_of_users, latent_factors);
    this->V = new matrix<int> (num_of_movies, latent_factors);
    this->latent_factors = latent_factors;
    this->eta = eta;
    this->num_of_movies = num_of_movies;
    this->num_of_users = num_of_users;
}

float SVD:grad_U(int Yij, int i, int j) {
    vector<int> Ui = row(this->U, i);
    vector<int> Vj = row(this->V, j);
    vector<int> temp1 = (this->reg * Ui) - Vj;
    float temp2 = Yij - this->mu * inner_prod(Ui, Vj);
    return this->eta * temp1;
}

float SVD:grad_V(int Yij, int i, int j) {
    vector<int> Ui = row(this->U, i);
    vector<int> Vj = row(this->V, j);
    vector<int> temp1;
}

vector<int *> SVD::predict() {

}

void SVD::fit() {

}

SVD::~SVD() {
    delete this->U;
    delete this->V;
}