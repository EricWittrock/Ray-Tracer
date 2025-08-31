NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe"
SRC_CPP = $(wildcard src/*.cpp)
SRC_CU = $(wildcard src/*.cu)


# compile everything in src
all:
	$(NVCC) $(SRC_CU) $(SRC_CPP) -o ./build/main
