NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe"
SRC_CPP = $(wildcard src/*.cpp)
SRC_CU = $(wildcard src/*.cu)
OUT = build/main.exe

all: build run

build: $(OUT)

$(OUT): $(SRC_CPP) $(SRC_CU)
	$(NVCC) $(SRC_CU) $(SRC_CPP) -o $(OUT)

run: $(OUT)
	$(OUT)

clean:
	del build\*
