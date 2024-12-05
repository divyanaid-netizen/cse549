# vector-size, warm-cache
TESTS += $(call test-name,256,yes)
TESTS += $(call test-name,1024,yes)
TESTS += $(call test-name,16384,yes)
TESTS += $(call test-name,65536,yes)
TESTS += $(call test-name,262144,yes)
TESTS += $(call test-name,1048576,yes)
TESTS += $(call test-name,4194304,yes)
TESTS += $(call test-name,16777216,yes)
