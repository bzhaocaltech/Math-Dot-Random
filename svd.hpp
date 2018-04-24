/* A basic matrix factorization model */

#include "model.hpp"

class SVD : public Model {
    private:
        /* The two matrices of latent factors
            * U is a num_users * latent_factors
            * V is a num_movies * latent_factors */
        int** U;
        int** V;
        // Num of latent factors
        int latent_factors;
        // Num of users and movies
        int num_users;
        int num_movies;
        // Learning rate
        int eta;

        float regularized_error(vector<int*>* data);
        
        float grad_U(int i, int j);
        float grad_V(int i, int j);
        float grad_A(int i, int j);
        float grad_B(int i, int j);
    public:
        /* Constructor for SVD */
        SVD(int latent_factors, int eta,
            int num_of_movies = NUM_MOVIES, int num_of_users = NUM_USERS);

        /* Given a list of x values in the form of (user, movie, time) predicts
            * the rating */
        vector<float>* predict(vector<int*>* x);

        /* Given a list of x values in the form of (user, movie, time, rating)
            * fits the model */
        void fit(vector<int*>* x);

        /* Destructor for SVD */
        ~SVD();
}
