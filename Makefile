NVCC = nvcc
CUDAFLAGS = -lcuda -Xcompiler -fopenmp -Xcompiler -std=c++14 -arch=compute_75 -code=sm_75 -O3 -g
LIBDIRS = -I.


src = $(wildcard src/*.cu)
obj = $(src:src/*.cu = .o)

main: $(obj)
	$(NVCC) $(CUDAFLAGS) -o build/$@ $^ $(LIBDIRS)


clean:
	rm -f build/main build/*.o
	