
MD_ND:
	g++ -std=c++26 -O3 -march=native -fopenmp -o MD_ND MD_ND.cpp -Wall -Wextra -Wpedantic

clean:
	rm MD_ND
