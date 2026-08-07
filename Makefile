CXX      ?= g++
CXXFLAGS ?= -O2 -march=native -mavx2 -mfma -std=c++17

BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
BLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)

INCLUDES := -Iinclude $(BLAS_CFLAGS)

BUILD := build
BIN   := $(BUILD)/cpu_example

SRCS := src/cpu/nuggem_cpu.cpp examples/cpu_example.cpp
OBJS := $(SRCS:%.cpp=$(BUILD)/%.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all run clean compdb

all: $(BIN)

compdb:
	$(MAKE) clean
	bear -- $(MAKE)

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(BLAS_LIBS)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

run: $(BIN)
	./$(BIN)

clean:
	rm -rf $(BUILD)

-include $(DEPS)
