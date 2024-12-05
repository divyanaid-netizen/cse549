#########################################################
# CHANGE ME: TESTS                                      #
# TESTS += $(call test-name,[buffer-size],[warm-cache]) #
#########################################################
#MSIZE = 524288
MSIZE = 16384
TESTS += $(call test-name,14,16384,$(MSIZE),no)
#TESTS += $(call test-name,16,8,$(MSIZE2),no)
	