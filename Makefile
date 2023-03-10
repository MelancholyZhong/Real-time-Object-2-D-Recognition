CC = g++
CFLAGS = -g -Wall -std=c++11 
CAM_PLIST_FLAG = -sectcreate __TEXT __info_plist Info.plist
PRODUCTS = vidDisplay test


OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

vidDisplay : vidDisplay.cpp filters.cpp csv_util.cpp match.cpp fetchFeature.cpp
	$(CC) $(CFLAGS) $(CAM_PLIST_FLAG) -o $@ $^ $(LIBS) 

test : main.cpp fetchFeature.cpp
	$(CC) $(CFLAGS) $(CAM_PLIST_FLAG) -o $@ $^ $(LIBS) 

.PHONY: clean

clean : 
	rm -r -f *.o *~ $(PRODUCTS) *.dSYM

