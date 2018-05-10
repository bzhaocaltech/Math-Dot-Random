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
    // constructor to create Vector object
    Vector(int length);
    void update_element(int idx, float val);
    float at(int idx);
    ~Vector();
};
