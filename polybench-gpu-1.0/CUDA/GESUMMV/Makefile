include ../make.config

EXECUTABLE := gesummv.exe
CUFILES := gesummv.cu

NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include

all:
	$(NVCC) -O2 $(NVCC_FLAGS) $(CUDA_LIB_DIR) $(GENCODE_FLAGS) ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe

