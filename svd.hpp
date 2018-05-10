/* A basic matrix factorization model */

#include "model.hpp"
#include "matrix.hpp"
#include "vector.hpp"

class SVD : public Model {
    private:
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

        void grad_U(struct dataset* ds);
        void grad_V(struct dataset* ds);
        void grad_a(struct dataset* ds);
        void grad_b(struct dataset* ds);
    public:
        /* Constructor for SVD */
        SVD(int latent_factors, float eta, float reg, int num_users = NUM_USERS,
            int num_movies = NUM_MOVIES);

        /* Given a list of x values in the form of (user, movie, time) predicts
            * the rating */
        std::vector<float>* predict(struct dataset* dataset);

        /* Given a list of x values in the form of (user, movie, time, rating)
         * fits the model */
        void fit(struct dataset* dataset, int epochs);

        /* Destructor for SVD */
        ~SVD();
};
