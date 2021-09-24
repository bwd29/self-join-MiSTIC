NVCC = nvcc
CUDAFLAGS = -lcuda -Xcompiler -fopenmp -Xcompiler -std=c++14 -arch=compute_60 -code=sm_60 -O3
LIBDIRS = -I.


src = $(wildcard src/*.cu)
obj = $(src:src/*.cu = .o)

main: $(obj)
	$(NVCC) $(CUDAFLAGS) -o build/$@ $^ $(LIBDIRS)

.PHONY: clean
clean:
	rm -f build/main build/*.o
	