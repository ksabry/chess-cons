# nvcc
NVCC      := nvcc
NVCCFLAGS := --std=c++17
# NVCCFLAGS += -O0
NVCCFLAGS += -O3
NVCCFLAGS += -g -G
NVCCFLAGS += --include-path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include"
NVCCFLAGS += --library-path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64"
NVCCFLAGS += --library cudart,cublas,cudnn

# ld
LDFLAGS  :=
LDLIBS   :=

SOURCE := source

# dep
C_HDRS   := $(wildcard $(SOURCE)/*.h)
CXX_SRCS := $(wildcard $(SOURCE)/*.cpp)
CU_SRCS  := $(wildcard $(SOURCE)/*.cu)

DEP        := $(C_HDRS) $(CXX_SRCS) $(CU_SRCS)

# build
BUILD_DIR := obj
OBJS      :=
OBJS      += $(patsubst source/%.cpp,$(BUILD_DIR)/%.obj,$(CXX_SRCS))
OBJS      += $(patsubst source/%.cu,$(BUILD_DIR)/%.obj,$(CU_SRCS))

BIN_DIR   := bin
BIN       := chess.exe

# $(info $$OBJS is [${OBJS}])

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

%.cpp: %.h

%.cu: %.h

$(BIN): $(BUILD_DIR) $(BIN_DIR) $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(BIN_DIR)/$@ $(OBJS) $(LDFLAGS) $(LDLIBS)

test: all

clean:
	rm -rf $(BUILD_DIR)