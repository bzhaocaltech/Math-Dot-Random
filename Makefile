CXX = g++
CXXFLAGS = -std=c++14 -Wall -g -DNDEBUG -DBOOST_UBLAS_NDEBUG -pthread -O3
BOOSTSERIALIZE = -lboost_serialization
.PHONY: clean
EXECUTABLES = load_data run_mean_model run_svd run_svdpp run_knn run_time_knn tune_time_knn
LOAD_DATA_DEP = load_data.o serialize.o
MEAN_MODEL_DEP = mean_model.o serialize.o run_mean_model.o model.o output.o
SVD_DEP = matrix.o vector.o svd.o run_svd.o model.o serialize.o output.o
SVDPP_DEP = matrix.o vector.o svd.o serialize.o output.o model.o svdpp.o run_svdpp.o
KNN_DEP = matrix.o knn.o serialize.o output.o model.o run_knn.o
TIME_KNN_DEP = matrix.o knn.o serialize.o output.o model.o time_knn.o run_time_knn.o
TUNE_TIME_KNN_DEP = matrix.o knn.o serialize.o output.o model.o time_knn.o tune_time_knn.o

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

svdpp.o: svd.o svdpp.cpp svdpp.hpp
	$(CXX) $(CXXFLAGS) -c svdpp.cpp

knn.o: knn.cpp knn.hpp model.o
	$(CXX) $(CXXFLAGS) -c knn.cpp

time_knn.o: time_knn.cpp knn.cpp time_knn.hpp
	$(CXX) $(CXXFLAGS) -c time_knn.cpp

matrix.o: matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp

run_mean_model: $(MEAN_MODEL_DEP)
	$(CXX) $(CXXFLAGS) $(MEAN_MODEL_DEP) -o run_mean_model $(BOOSTSERIALIZE)

run_svd: $(SVD_DEP)
	$(CXX) $(CXXFLAGS) $(SVD_DEP) -o run_svd $(BOOSTSERIALIZE)

run_svdpp: $(SVDPP_DEP)
	$(CXX) $(CXXFLAGS) $(SVDPP_DEP) -o run_svdpp $(BOOSTSERIALIZE)

run_knn: $(KNN_DEP)
	$(CXX) $(CXXFLAGS) $(KNN_DEP) -o run_knn $(BOOSTSERIALIZE)

run_time_knn: $(TIME_KNN_DEP)
	$(CXX) $(CXXFLAGS) $(TIME_KNN_DEP) -o run_time_knn $(BOOSTSERIALIZE)

tune_time_knn: $(TUNE_TIME_KNN_DEP)
	$(CXX) $(CXXFLAGS) $(TUNE_TIME_KNN_DEP) -o tune_time_knn $(BOOSTSERIALIZE)

clean:
	$(RM) $(EXECUTABLES) *.o results.dta
