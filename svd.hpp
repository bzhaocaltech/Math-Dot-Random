/* A basic matrix factorization model */

#include "model.hpp"
#include "matrix.hpp"

using namespace std;

class SVD : public Model {
    private:
        /* The two matrices of latent factors
            * U is a num_users * latent_factors
            * V is a num_movies * latent_factors */
        Matrix* U;
        Matrix* V;
        // The user biases
        float* a;
        // The movie biases
        float* b;
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

        void grad_U(int Yij, int i, int j);
        void grad_V(int Yij, int i, int j);
        void grad_a(int Yij, int i, int j);
        void grad_b(int Yij, int i, int j);

        /* Serializes the model into a given file */
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
        vector<float>* predict(struct dataset* dataset);

        /* Given a list of x values in the form of (user, movie, time, rating)
         * fits the model */
        void fit(struct dataset* dataset, int epochs);

        /* Destructor for SVD */
        ~SVD();
};
