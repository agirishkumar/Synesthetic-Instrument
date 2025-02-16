CC = nvcc
CUDA_PATH = /usr/local/cuda

# Get GLib compilation flags
GLIB_CFLAGS = $(shell pkg-config --cflags glib-2.0)
GLIB_LIBS = $(shell pkg-config --libs glib-2.0)

# Warning suppressions for CUDA and GLM
WARNING_FLAGS = -Xcudafe --diag_suppress=20012 \
                -Xcudafe --diag_suppress=20011 \
                -Wno-deprecated-declarations

CFLAGS = -arch=sm_86 \
         -std=c++17 \
         -O3 \
         -I./include \
         -I$(CUDA_PATH)/include \
         -I$(CUDA_PATH)/samples/common/inc \
         -I/usr/include/GL \
         $(GLIB_CFLAGS) \
         -L$(CUDA_PATH)/lib64 \
         -L/usr/lib/x86_64-linux-gnu \
         $(WARNING_FLAGS) \
         -D_REENTRANT \
         -D_GNU_SOURCE

# GLM is header-only, so we remove it from LIBS
LIBS = -lportmidi -lporttime -lsmf -lGL -lGLEW -lglfw -lcudart $(GLIB_LIBS)

SRC_DIR = src
KERNEL_DIR = kernels
BIN_DIR = bin
BUILD_DIR = build

# Source files
SOURCES = $(SRC_DIR)/main.cu \
          $(SRC_DIR)/midi_processor.cu \
          $(KERNEL_DIR)/particle_kernels.cu

# Object files
OBJECTS = $(SOURCES:%.cu=$(BUILD_DIR)/%.o)

# Make sure the build directory exists
$(shell mkdir -p $(BUILD_DIR)/$(SRC_DIR) $(BUILD_DIR)/$(KERNEL_DIR))

all: $(BIN_DIR)/synesthesia

$(BIN_DIR)/synesthesia: $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LIBS)

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)

.PHONY: all clean