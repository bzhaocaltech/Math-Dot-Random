/* A basic matrix factorization model */

#include "model.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

using namespace boost::numeric::ublas;

class SVD : public Model {
    private:
        /* The two matrices of latent factors
            * U is a num_users * latent_factors
            * V is a num_movies * latent_factors */
        matrix<float> *U;
        matrix<float> *V;
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
        float mu;

        boost::numeric::ublas::vector<float> grad_U(int Yij, int i, int j);
        boost::numeric::ublas::vector<float> grad_V(int Yij, int i, int j);
        float grad_a(int Yij, int i, int j);
        float grad_b(int Yij, int i, int j);
    public:
        /* Constructor for SVD */
        SVD(int latent_factors, float eta, float reg,
            int num_movies = NUM_MOVIES, int num_users = NUM_USERS);

        /* Given a list of x values in the form of (user, movie, time) predicts
            * the rating */
        std::vector<float>* predict(std::vector<int*>* x);

        /* Given a list of x values in the form of (user, movie, time, rating)
            * fits the model */
        void fit(std::vector<int*>* x, int epochs);

        void fit(std::vector<int*>* x);

        /* Destructor for SVD */
        ~SVD();
};
