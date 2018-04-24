/* A basic matrix factorization model */

#include "svd.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace boost::numeric::ublas;

/* Constructor for SVD */
SVD::SVD(int latent_factors, float eta, float reg, int num_users, int num_movies) {
    this->U (num_users, latent_factors);
    this->V (num_movies, latent_factors);
    this->a = new float[num_users];
    this->b = new float[num_movies];
    this->latent_factors = latent_factors;
    this->eta = eta;
    this->reg = reg;
    this->num_movies = num_movies;
    this->num_users = num_users;
}

/* Takes as input the actual rating for the ith user and the jth movie.
 * Returns the gradient of the regularized loss function with respect to
 * Ui multiplied by eta */
boost::numeric::ublas::vector<float> SVD::grad_U(int Yij, int i, int j) {
    boost::numeric::ublas::vector<float> Ui = row(this->U, i);
    boost::numeric::ublas::vector<float> Vj = row(this->V, j);
    boost::numeric::ublas::vector<float> temp1 = (this->reg * Ui);
    float temp2 = Yij - this->mu - inner_prod(Ui, Vj) - a[i] - b[j];
    return this->eta * (temp1 - Vj * temp2);
}

/* Takes as input the actual rating for the ith user and the jth movie.
 * Returns the gradient of the regularized loss function with respect to
 * Vj multiplied by eta */
boost::numeric::ublas::vector<float> SVD::grad_V(int Yij, int i, int j) {
    boost::numeric::ublas::vector<float> Ui = row(this->U, i);
    boost::numeric::ublas::vector<float> Vj = row(this->V, j);
    boost::numeric::ublas::vector<float> temp1 = (this->reg * Vj);
    float temp2 = Yij - this->mu - inner_prod(Ui, Vj) - a[i] - b[j];
    return this->eta * (temp1 - Ui * temp2);
}

/* Same input as other grad functions. Returns the graident of the regularized
 * loss function with respect to ai multiplied by eta. */
float SVD::grad_a(int Yij, int i, int j) {
    boost::numeric::ublas::vector<float> Ui = row(this->U, i);
    boost::numeric::ublas::vector<float> Vj = row(this->V, j);
    float temp1 = this->reg * a[i];
    float temp2 = Yij - this->mu - inner_prod(Ui, Vj) - a[i] - b[j];
    return this->eta * (temp1 - temp2);
}

/* Same input as other grad functions. Returns the graident of the regularized
 * loss function with respect to bj multiplied by eta. */
float SVD::grad_b(int Yij, int i, int j) {
    boost::numeric::ublas::vector<float> Ui = row(this->U, i);
    boost::numeric::ublas::vector<float> Vj = row(this->V, j);
    float temp1 = this->reg * b[j];
    float temp2 = Yij - this->mu - inner_prod(Ui, Vj) - a[i] - b[j];
    return this->eta * (temp1 - temp2);
}

/* Given a list of x values in the form of (user, movie, time) predicts
    * the rating */
std::vector<float>* SVD::predict(std::vector<int*>* x) {
    std::vector<float>* predictions = new std::vector<float>();
    for (unsigned int i = 0; i < x->size(); i++) {
        int* data = x->at(i);
        int user = data[0];
        int movie = data[1];
        boost::numeric::ublas::vector<float> Ui = row(this->U, user);
        boost::numeric::ublas::vector<float> Vj = row(this->V, movie);
        float p = inner_prod(Ui, Vj) + this->a[user] + this->b[movie] + this->mu;
        predictions->push_back(p);
    }
    return predictions;
}

void SVD::fit(std::vector<int*>* x, int epochs) {
    fprintf(stderr, "Fitting the data of size %i\n", (int) x->size());

    // Calculate the global bias
    fprintf(stderr, "Calculating the global bias\n");
    for (unsigned int i = 0; i < x->size(); i++) {
        int rating = x->at(i)[3];
        this->mu += rating;
    }
    this->mu /= (float) x->size();
    fprintf(stderr, "Global bias was %f\n", this->mu);

    // Initialize U, V, a, b randomly
    fprintf(stderr, "Randomly initializing matrices\n");
    for (int i = 0; i < this->num_users; i++) {
        for (int j = 0; j < this->latent_factors; j++) {
            float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
            this->U (i, j) = random;
        }
    }
    for (int i = 0; i < this->num_movies; i++) {
        for (int j = 0; j < this->latent_factors; j++) {
            float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
            this->V (i, j) = random;
        }
    }
    for (int i = 0; i < this->num_users; i++) {
        float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
        this->a[i] = random;
    }
    for (int j = 0; j < this->num_movies; j++) {
        float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
        this->b[j] = random;
    }
    fprintf(stderr, "Matrices randomly initialized\n");

    for (int curr_epoch = 0; curr_epoch < epochs; curr_epoch++) {
        fprintf(stderr, "Running epoch %i", curr_epoch);
        for (unsigned int i = 0; i < x->size(); i++) {
            int* data = x->at(i);
            int user = data[0];
            int movie = data[1];
            int rating = data[3];
            row(this->U, user) -= grad_U(rating, user, movie);
            row(this->V, movie) -= grad_V(rating, user, movie);
            row(this->a, user) -= grad_a(rating, user, movie);
            row(this->b, movie) -= grad_b(rating, user, movie);
            if (i % 3000000 == 0) {
                fprintf(stderr, ".");
            }
        }
        fprintf(stderr, "\n");
    }
}

SVD::~SVD() {
    delete this->a;
    delete this->b;
}
