CXX = g++
CXXFLAGS = -std=c++14 -Wall -g -DNDEBUG -DBOOST_UBLAS_NDEBUG -pthread -O3
BOOSTSERIALIZE = -lboost_serialization
.PHONY: clean
EXECUTABLES = load_data run_mean_model run_svd
LOAD_DATA_DEP = load_data.o serialize.o
MEAN_MODEL_DEP = mean_model.o serialize.o run_mean_model.o model.o output.o
SVD_DEP = matrix.o vector.o svd.o run_svd.o model.o serialize.o output.o

all: $(EXECUTABLES)

load_data: $(LOAD_DATA_DEP)
	$(CXX) $(CXXFLAGS) $(BOOSTROOT) $(LOAD_DATA_DEP) -o load_data $(BOOSTSERIALIZE)

serialize.o: serialize.cpp serialize.hpp data.hpp
	$(CXX) $(CXXFLAGS) -c serialize.cpp

load_data.o: load_data.cpp
	$(CXX) $(CXXFLAGS) -c load_data.cpp

model.o: model.hpp model.cpp
	$(CXX) $(CXXFLAGS) -c model.cpp

output.o: output.hpp output.cpp
	$(CXX) $(CXXFLAGS) -c output.cpp

mean_model.o: model.o mean_model.cpp mean_model.hpp
	$(CXX) $(CXXFLAGS) -c mean_model.cpp

svd.o: model.o svd.cpp svd.hpp
	$(CXX) $(CXXFLAGS) -c svd.cpp

matrix.o: matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp

run_mean_model: $(MEAN_MODEL_DEP)
	$(CXX) $(CXXFLAGS) $(MEAN_MODEL_DEP) -o run_mean_model $(BOOSTSERIALIZE)

run_svd: $(SVD_DEP)
	$(CXX) $(CXXFLAGS) $(SVD_DEP) -o run_svd $(BOOSTSERIALIZE)

clean:
	$(RM) $(EXECUTABLES) *.o results.dta
