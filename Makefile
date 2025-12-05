NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe"
SRC_CPP = $(wildcard src/*.cpp)
SRC_CU = $(wildcard src/*.cu)
OUT = build/main.exe

NVCC_FLAGS_DEBUG = -O0 -g -Xptxas -O0 -Xcompiler "/Od /W4 /Zi"
NVCC_FLAGS = -rdc=true

all: build run

build: $(OUT)

$(OUT): $(SRC_CPP) $(SRC_CU)
	$(NVCC) $(NVCC_FLAGS) $(SRC_CU) $(SRC_CPP) -o $(OUT)

run: $(OUT)
	$(OUT)

clean:
	del build\*
