CXX ?= c++
CXXFLAGS ?= -std=c++20 -O3 -march=native -pthread -Wall -Wextra
LDFLAGS ?= -pthread

TARGET := join_ht_build_bench
SRC := bench.cpp

.PHONY: all clean run quick

all: $(TARGET)

$(TARGET): $(SRC) platform.h hashes.h zipf.h latch.h umbra.h ch_table.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

quick: $(TARGET)
	./$(TARGET) --quick

run: $(TARGET)
	./$(TARGET)
