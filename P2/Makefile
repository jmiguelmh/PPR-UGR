# Compilador
CXX = mpicxx
CFLAGS = -O3

all: p2_1d p2_2d

p2_1d: p2_1d.cpp
	$(CXX) $(CFLAGS) p2_1d.cpp -o p2_1d

p2_2d: p2_2d.cpp
	$(CXX) $(CFLAGS) p2_2d.cpp -o p2_2d

clean:
	rm -f p2_1d p2_2d