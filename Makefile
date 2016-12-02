all: main

main: cpu
	nvcc nodes_cuda.cu -o "gpu"
cpu:
	g++ -std=c++11 nodes.cpp -o "cpu"

clean:
	rm gpu cpu 
