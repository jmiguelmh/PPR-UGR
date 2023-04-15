# CUDA code generation flags

GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SMXX    := -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   ?=  $(GENCODE_SM60)

floyd: floyd.cu Graph.cc Graph.h
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS)   floyd.cu Graph.cc -o floyd

clean:
	rm -f floyd *.o.

