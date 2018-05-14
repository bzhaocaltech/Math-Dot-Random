// Thread-safe vector implementation. Probably a lot of overhead...

#include <vector>
#include <mutex>

class Vector {
private:
    // the actual data
    float* vector;
    // locks to keep vector implementation thread safe
    std::vector<std::mutex*>* vector_locks;
public:
    // constructor to create a empty Vector object
    Vector(int length);
    // constructor to create a vector object from a float*
    Vector(int length, float* vector);
    // Returns the float* vector inside of vector
    float* get_vector();
    void update_element(int idx, float val);
    float at(int idx);
    ~Vector();
};
