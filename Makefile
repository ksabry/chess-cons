MODE      := gpu
BUILD_DIR := obj
BIN_DIR   := bin
BIN       := chess.exe

# nvcc
NVCC      := nvcc
NVCCFLAGS := --std=c++17
# NVCCFLAGS += -O0
NVCCFLAGS += -O3
NVCCFLAGS += -g -G
ifeq ($(MODE),gpu)
	NVCCFLAGS += --include-path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include"
	NVCCFLAGS += --library-path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64"
	NVCCFLAGS += --library cudart,cublas,cudnn
else
endif

# ld
LDFLAGS  :=
LDLIBS   :=

SOURCE := source

# dep
C_HDRS        := $(wildcard $(SOURCE)/*.h)
CU_SRCS       := $(wildcard $(SOURCE)/*.cu)
CXX_CPU_SRCS  := $(wildcard $(SOURCE)/*.cpu.cpp)
CXX_GPU_SRCS  := $(wildcard $(SOURCE)/*.gpu.cpp)
CXX_ALL_SRCS  := $(wildcard $(SOURCE)/*.cpp)
CXX_EITHER_SRCS := $(filter-out $(CXX_CPU_SRCS) $(CXX_GPU_SRCS),$(CXX_ALL_SRCS))

ifeq ($(MODE),gpu)
	DEP  := $(C_HDRS) $(CXX_EITHER_SRCS) $(CXX_GPU_SRCS) $(CU_SRCS)
	OBJS :=
	OBJS += $(patsubst source/%.cpp,$(BUILD_DIR)/%.obj,$(CXX_EITHER_SRCS))
	OBJS += $(patsubst source/%.cpp,$(BUILD_DIR)/%.obj,$(CXX_GPU_SRCS))
	OBJS += $(patsubst source/%.cu,$(BUILD_DIR)/%.obj,$(CU_SRCS))
else
	DEP  := $(C_HDRS) $(CXX_EITHER_SRCS) $(CXX_CPU_SRCS)
	OBJS :=
	OBJS += $(patsubst source/%.cpp,$(BUILD_DIR)/%.obj,$(CXX_EITHER_SRCS))
	OBJS += $(patsubst source/%.cpp,$(BUILD_DIR)/%.obj,$(CXX_CPU_SRCS))
endif

$(info $$DEP is [${DEP}])
$(info $$OBJS is [${OBJS}])

#########
# rules #
#########

.PHONY: pre test* clean

all: $(BIN)

$(BUILD_DIR):
	mkdir $@

$(BIN_DIR):
	mkdir $@

$(BUILD_DIR)/%.obj: $(SOURCE)/%.cpp
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

%.c: %.h

%.cpu.cpp: %.h

%.gpu.cpp: %.h

%.cpp: %.h

%.cu: %.h

$(BIN): $(BUILD_DIR) $(BIN_DIR) $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(BIN_DIR)/$@ $(OBJS) $(LDFLAGS) $(LDLIBS)

test: all

clean:
	rm -rf $(BUILD_DIR)