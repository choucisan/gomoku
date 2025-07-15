# Makefile
CXX = c++

CXXFLAGS = -std=c++17 -Wall -O2


SRCS = train.cpp gomoku/gomoku.cpp model/mlp.cpp

TARGET = train

all: $(TARGET)

$(TARGET):
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

.PHONY: all clean