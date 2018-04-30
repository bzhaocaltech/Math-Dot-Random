/* A basic matrix factorization model */

#include "svd.hpp"
#include <stdlib.h>
#include <stdio.h>

/* Constructor for SVD */
SVD::SVD(int latent_factors, float eta, float reg, int num_users, int num_movies) {
    this->U = new Matrix(num_users, latent_factors);
    this->V = new Matrix(num_movies, latent_factors);
    this->a = new float[num_users];
    this->b = new float[num_movies];
    this->latent_factors = latent_factors;
    this->eta = eta;
    this->reg = reg;
    this->num_movies = num_movies;
    this->num_users = num_users;
    printf("Creating SVD with %i latent factors, learning rate %f, and reg %f \n",
          this->latent_factors, this->eta, this->reg);
}

/* Takes as input the actual rating for the ith user and the jth movie.
 * Returns the gradient of the regularized loss function with respect to
 * Ui multiplied by eta */
void SVD::grad_U(int Yij, int i, int j) {
    float* Ui = this->U->row(i);
    float* Vj = this->V->row(j);
    float* temp1 = scalar_vec_prod(this->reg, Ui, this->latent_factors);
    float temp2 = ((float) Yij) - this->mu - dot_prod(Ui, Vj, this->latent_factors)
                      - this->a[i] - this->b[j];
    float* temp3 = scalar_vec_prod(temp2, Vj, this->latent_factors);
    float* temp4 = vec_sub(temp1, temp3, this->latent_factors);
    delete temp1;
    delete temp3;
    float* grad_u = scalar_vec_prod(this->eta, temp4, this->latent_factors);
    delete temp4;
    float* new_u = vec_sub(Ui, grad_u, this->latent_factors);
    delete grad_u;
    this->U->update_row(i, new_u);
}

/* Takes as input the actual rating for the ith user and the jth movie.
 * Returns the gradient of the regularized loss function with respect to
 * Vj multiplied by eta */
void SVD::grad_V(int Yij, int i, int j) {
    float* Ui = this->U->row(i);
    float* Vj = this->V->row(j);
    float* temp1 = scalar_vec_prod(this->reg, Vj, this->latent_factors);
    float temp2 = ((float) Yij) - this->mu - dot_prod(Ui, Vj, this->latent_factors)
                      - this->a[i] - this->b[j];
    float* temp3 = scalar_vec_prod(temp2, Ui, this->latent_factors);
    float* temp4 = vec_sub(temp1, temp3, this->latent_factors);
    delete temp1;
    delete temp3;
    float* grad_v = scalar_vec_prod(this->eta, temp4, this->latent_factors);
    delete temp4;
    float* new_v = vec_sub(Vj, grad_v, this->latent_factors);
    delete grad_v;
    this->V->update_row(j, new_v);
}

/* Same input as other grad functions. Returns the graident of the regularized
 * loss function with respect to ai multiplied by eta. */
void SVD::grad_a(int Yij, int i, int j) {
    float* Ui = this->U->row(i);
    float* Vj = this->V->row(j);
    float temp1 = this->reg * this->a[i];
    float temp2 = Yij - this->mu - dot_prod(Ui, Vj, this->latent_factors)
                  - this->a[i] - this->b[j];
    float value = this->eta * (temp1 - temp2);
    a[i] -= value;
}

/* Same input as other grad functions. Returns the graident of the regularized
 * loss function with respect to bj multiplied by eta. */
void SVD::grad_b(int Yij, int i, int j) {
    float* Ui = this->U->row(i);
    float* Vj = this->V->row(j);
    float temp1 = this->reg * this->b[j];
    float temp2 = Yij - this->mu - dot_prod(Ui, Vj, this->latent_factors)
                  - this->a[i] - this->b[j];
    float value = this->eta * (temp1 - temp2);
    b[j] -= value;
}

/* Given a list of x values in the form of (user, movie, time) predicts
    * the rating */
vector<float>* SVD::predict(std::vector<int*>* x) {
    vector<float>* predictions = new vector<float>();
    for (unsigned int i = 0; i < x->size(); i++) {
        int* data = x->at(i);
        int user = data[0];
        int movie = data[1];
        float* Ui = this->U->row(user);
        float* Vj = this->V->row(movie);
        float p = dot_prod(Ui, Vj, this->latent_factors) + this->a[user]
                  + this->b[movie] + this->mu;
        predictions->push_back(p);
    }
    return predictions;
}

/* Given a list of x values in the form of (user, movie, time, rating)
 * fits the model */
void SVD::fit(std::vector<int*>* x, int epochs) {
    fprintf(stderr, "Fitting the data of size %i\n", (int) x->size());

    // Calculate the global bias
    this->mu = 0;
    fprintf(stderr, "Calculating the global bias\n");
    for (unsigned int i = 0; i < x->size(); i++) {
        int rating = x->at(i)[3];
        this->mu += (double) rating;
    }
    this->mu /= (double) x->size();
    fprintf(stderr, "Global bias was %f\n", this->mu);

    // Initialize U, V, a, b randomly
    fprintf(stderr, "Randomly initializing matrices\n");
    for (int i = 0; i < this->num_users; i++) {
        for (int j = 0; j < this->latent_factors; j++) {
            float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
            this->U->set_val(i, j, random);
        }
    }
    for (int i = 0; i < this->num_movies; i++) {
        for (int j = 0; j < this->latent_factors; j++) {
            float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
            this->V->set_val(i, j, random);
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
        fprintf(stderr, "Running epoch %i", curr_epoch + 1);
        for (unsigned int i = 0; i < x->size(); i++) {
            int user = x->at(i)[0];
            int movie = x->at(i)[1];
            int rating = x->at(i)[3];
            grad_V(rating, user, movie);
            grad_U(rating, user, movie);
            grad_a(rating, user, movie);
            grad_b(rating, user, movie);
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
    delete this->U;
    delete this->V;
}
