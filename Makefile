CXX = g++
CXXFLAGS = -std=c++14 -Wall -g
BOOSTROOT = -L/usr/local/lib
BOOSTSERIALIZE = -lboost_serialization
.PHONY: clean
TOCLEAN = load_data *.o

load_data: load_data.o
	$(CXX) $(CXXFLAGS) $(BOOSTROOT) -static load_data.o serialize.o -o load_data $(BOOSTSERIALIZE)

serialize.o: serialize.cpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c serialize.cpp

load_data.o: load_data.cpp serialize.o
	$(CXX) $(CXXFLAGS) -c load_data.cpp

clean:
	$(RM) $(TOCLEAN)
