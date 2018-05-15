/* A basic matrix factorization model */

#include "model.hpp"
#include "matrix.hpp"
#include "vector.hpp"

class SVD : public Model {
    protected:
        /* The two matrices of latent factors
            * U is a num_users * latent_factors
            * V is a num_movies * latent_factors */
        Matrix* U;
        Matrix* V;
        // The user biases
        Vector* a;
        // The movie biases
        Vector* b;
        // Num of latent factors
        int latent_factors;
        // Num of users and movies
        int num_users;
        int num_movies;
        // Learning rate
        float eta;
        // Regularization factor
        float reg;
        // Global bias
        double mu;

        /* Run a gradient on part of the dataset */
        void grad_part(struct dataset* ds, bool track_progress);
        void grad_U(struct data d);
        void grad_V(struct data d);
        void grad_a(struct data d);
        void grad_b(struct data d);

    public:
        /* Constructor for SVD */
        SVD(int latent_factors, float eta, float reg, int num_users = NUM_USERS,
            int num_movies = NUM_MOVIES);

        /* Constructs a SVD from a file
         * NOTE: Please remeber the .ser file extension */
        SVD(string file);

        /* Serializes the model into a given file
         * NOTE: Please remeber the .ser file extension */
        void serialize(string file);

        /* Given a list of x values in the form of (user, movie, time) predicts
            * the rating */
        std::vector<float>* predict(struct dataset* dataset);

        /* Given a list of x values in the form of (user, movie, time, rating)
         * fits the model */
        void fit(struct dataset* dataset, int epochs, int num_threads = 10);

        /* Destructor for SVD */
        ~SVD();
};
