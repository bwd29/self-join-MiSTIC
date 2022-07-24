NVCC = nvcc
CUDAFLAGS = -lcuda -Xcompiler -fopenmp -arch=compute_61 -code=sm_61 -O3
LIBDIRS = -I.


build/main: build/main.o build/launcher.o build/kernel.o build/nodes.o build/tree.o build/utils.o 
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -o build/main build/main.o build/launcher.o build/kernel.o build/nodes.o build/tree.o build/utils.o

build/main.o: src/main.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -c -o build/main.o src/main.cu -lm

build/launcher.o: src/launcher.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -c -o build/launcher.o src/launcher.cu -lm

build/kernel.o: src/kernel.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -Xcompiler -std=c++03 -c -o build/kernel.o src/kernel.cu -lm

build/nodes.o: src/nodes.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -c -o build/nodes.o src/nodes.cu -lm

build/tree.o: src/tree.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -c -o build/tree.o src/tree.cu -lm

build/utils.o: src/utils.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -c -o build/utils.o src/utils.cu -lm

clean:
	rm -f build/main build/*.o
	