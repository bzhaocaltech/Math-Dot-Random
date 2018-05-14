/* A basic matrix factorization model */

#include "svd.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

/* Constructor for SVD */
SVD::SVD(int latent_factors, float eta, float reg, int num_users, int num_movies) {
    this->U = new Matrix(num_users, latent_factors);
    this->V = new Matrix(num_movies, latent_factors);
    this->a = new Vector(num_users);
    this->b = new Vector(num_movies);
    this->latent_factors = latent_factors;
    this->eta = eta;
    this->reg = reg;
    this->num_movies = num_movies;
    this->num_users = num_users;
    printf("Creating SVD with %i latent factors, learning rate %f, and reg %f \n",
          this->latent_factors, this->eta, this->reg);
}

/* Constructs a SVD from a file */
SVD::SVD(string file) {
    fprintf(stderr, "Unserializing SVD from %s \n", file.c_str());
    ifstream ifs(file);
    boost::archive::binary_iarchive ia(ifs);

    /* Unserialize the parameters of this SVD */
    ia & latent_factors;
    ia & eta;
    ia & reg;
    ia & num_movies;
    ia & num_users;
    ia & mu;

    /* Unserialize the U and V matrices */
    U = new Matrix(file.substr(0, file.length() - 4) + "_U_matrix.ser");
    V = new Matrix(file.substr(0, file.length() - 4) + "_V_matrix.ser");

    /* Unserialize a and b */
    float* a_float = new float[num_users];
    for (int i = 0; i < num_users; i++) {
        ia & a_float[i];
    }
    this->a = new Vector(num_users, a_float);
    float* b_float = new float[num_movies];
    for (int i = 0; i < num_movies; i++) {
        ia & b_float[i];
    }
    this->b = new Vector(num_movies, b_float);
}

/* Serializes the model into a given file  */
void SVD::serialize(string file) {
    fprintf(stderr, "Serializing SVD to %s \n", file.c_str());
    ofstream ofs(file);
    boost::archive::binary_oarchive oa(ofs);

    /* Serialize parameters of this SVD */
    oa & latent_factors;
    oa & eta;
    oa & reg;
    oa & num_movies;
    oa & num_users;
    oa & mu;

    /* Serialize the U and V matrices */
    U->serialize(file.substr(0, file.length() - 4) + "_U_matrix.ser");
    V->serialize(file.substr(0, file.length() - 4) + "_V_matrix.ser");

    /* Serialize a and b */
    float* vec_a = this->a->get_vector();
    for (int i = 0; i < num_users; i++) {
        oa & vec_a[i];
    }
    float* vec_b = this->b->get_vector();
    for (int i = 0; i < num_movies; i++) {
        oa & vec_b[i];
    }
}

/* Takes as input the actual rating for the ith user and the jth movie.
 * Returns the gradient of the regularized loss function with respect to
 * Ui multiplied by eta */
void SVD::grad_U(struct dataset* ds) {
    for (int n = 0; n < ds->size; n++) {
        int Yij = ds->data[n].rating;
        int i = ds->data[n].user;
        int j = ds->data[n].movie;
        float* Ui = this->U->row(i);
        float* Vj = this->V->row(j);
        float* temp1 = scalar_vec_prod(this->reg, Ui, this->latent_factors);
        float temp2 = ((float) Yij) - this->mu - dot_prod(Ui, Vj, this->latent_factors)
        - this->a->at(i) - this->b->at(j);
        float* temp3 = scalar_vec_prod(temp2, Vj, this->latent_factors);
        float* temp4 = vec_sub(temp1, temp3, this->latent_factors);
        delete temp1;
        delete temp3;
        float* grad_u = scalar_vec_prod(this->eta, temp4, this->latent_factors);
        delete temp4;
        float* new_u = vec_sub(Ui, grad_u, this->latent_factors);
        delete grad_u;
        this->U->update_row(i, new_u);
        if (n % 3000000 == 0) {
            fprintf(stderr, ".");
        }
    }
}

/* Takes as input the actual rating for the ith user and the jth movie.
 * Returns the gradient of the regularized loss function with respect to
 * Vj multiplied by eta */
void SVD::grad_V(struct dataset* ds) {
    for (int n = 0; n < ds->size; n++) {
        int Yij = ds->data[n].rating;
        int i = ds->data[n].user;
        int j = ds->data[n].movie;
        float* Ui = this->U->row(i);
        float* Vj = this->V->row(j);
        float* temp1 = scalar_vec_prod(this->reg, Vj, this->latent_factors);
        float temp2 = ((float) Yij) - this->mu - dot_prod(Ui, Vj, this->latent_factors)
                          - this->a->at(i) - this->b->at(j);
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
}

/* Same input as other grad functions. Returns the graident of the regularized
 * loss function with respect to ai multiplied by eta. */
void SVD::grad_a(struct dataset* ds) {
    for (int n = 0; n < ds->size; n++) {
        int Yij = ds->data[n].rating;
        int i = ds->data[n].user;
        int j = ds->data[n].movie;
        float* Ui = this->U->row(i);
        float* Vj = this->V->row(j);
        float temp1 = this->reg * this->a->at(i);
        float temp2 = Yij - this->mu - dot_prod(Ui, Vj, this->latent_factors)
                      - this->a->at(i) - this->b->at(j);
        float value = this->eta * (temp1 - temp2);
        this->a->update_element(i, this->a->at(i) - value);
    }
}

/* Same input as other grad functions. Returns the graident of the regularized
 * loss function with respect to bj multiplied by eta. */
void SVD::grad_b(struct dataset* ds) {
    for (int n = 0; n < ds->size; n++) {
        int Yij = ds->data[n].rating;
        int i = ds->data[n].user;
        int j = ds->data[n].movie;
        float* Ui = this->U->row(i);
        float* Vj = this->V->row(j);
        float temp1 = this->reg * this->b->at(j);
        float temp2 = Yij - this->mu - dot_prod(Ui, Vj, this->latent_factors)
                      - this->a->at(i) - this->b->at(j);
        float value = this->eta * (temp1 - temp2);
        this->b->update_element(j, this->b->at(j) - value);
    }
}

/* Given a list of x values in the form of (user, movie, time) predicts
    * the rating */
vector<float>* SVD::predict(struct dataset* dataset) {
    vector<float>* predictions = new vector<float>();
    for (int i = 0; i < dataset->size; i++) {
        struct data data = dataset->data[i];
        int user = data.user;
        int movie = data.movie;
        float* Ui = this->U->row(user);
        float* Vj = this->V->row(movie);
        float p = dot_prod(Ui, Vj, this->latent_factors) + this->a->at(user)
                  + this->b->at(movie) + this->mu;
        predictions->push_back(p);
    }
    return predictions;
}

/* Given a list of x values in the form of (user, movie, time, rating)
 * fits the model */
void SVD::fit(struct dataset* dataset, int epochs) {
    fprintf(stderr, "Fitting the data of size %i\n", dataset->size);

    // Calculate the global bias
    this->mu = 0;
    fprintf(stderr, "Calculating the global bias\n");
    for (int i = 0; i < dataset->size; i++) {
        int rating = dataset->data[i].rating;
        this->mu += (double) rating;
    }
    this->mu /= (double) dataset->size;
    fprintf(stderr, "Global bias was %f\n", this->mu);

    fprintf(stderr, "Running %i epochs\n", epochs);
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
        this->a->update_element(i, random);
    }
    for (int j = 0; j < this->num_movies; j++) {
        float random = (((float) rand()) / (float) RAND_MAX) - 0.5;
        this->b->update_element(j, random);
    }
    fprintf(stderr, "Matrices randomly initialized\n");

    for (int curr_epoch = 0; curr_epoch < epochs; curr_epoch++) {
        fprintf(stderr, "Running epoch %i", curr_epoch + 1);
        // for (int i = 0; i < dataset->size; i++) {
        //     struct data data = dataset->data[i];
        //     int user = data.user;
        //     int movie = data.movie;
        //     int rating = data.rating;
        //     grad_V(rating, user, movie);
        //     grad_U(rating, user, movie);
        //     grad_a(rating, user, movie);
        //     grad_b(rating, user, movie);
        //     if (i % 3000000 == 0) {
        //         fprintf(stderr, ".");
        //     }
        // }
        std::thread u_thread (&SVD::grad_U, this, dataset);
        std::thread v_thread (&SVD::grad_V, this, dataset);
        std::thread a_thread (&SVD::grad_a, this, dataset);
        std::thread b_thread (&SVD::grad_b, this, dataset);

        u_thread.join();
        v_thread.join();
        a_thread.join();
        b_thread.join();

        fprintf(stderr, "\n");
    }
}

SVD::~SVD() {
    delete this->a;
    delete this->b;
    delete this->U;
    delete this->V;
}
