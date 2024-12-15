HB_HAMMERBENCH_PATH = $(shell git rev-parse --show-toplevel)

.PHONY: all clean generate

# Default target
all: generate

# Call to generate a test name
test-name = sort-items_$(1)

# Call to get the buffer size from the test name
get-buffer-size = $(lastword $(subst _, ,$(filter sort-items%,$(subst __, ,$(1)))))

# Defines tests
TESTS =
include tests.mk
TESTS_DIRS = $(TESTS)

# Rule to generate parameters.mk in each test directory
$(addsuffix /parameters.mk, $(TESTS_DIRS)): %/parameters.mk:
	@echo "Creating $@"
	@mkdir -p $(dir $@)
	@touch $@
	@echo "test-name = $*" >> $@
	@echo "buffer-size = $(call get-buffer-size, $*)" >> $@

# Include common testbench definitions
include $(HB_HAMMERBENCH_PATH)/mk/testbench_common.mk

# Define buffer size for compilation
BUFFER_SIZE ?= 256 # Default buffer size for sorting

# Compilation flags
CFLAGS += -DARRAY_SIZE=$(BUFFER_SIZE)

# Clean rule
clean:
	rm -rf sort-items_*
