#pkg-config --libs opencv
#CXXFLAGS= -D_GLIBCXX_USE_CXX11_ABI=0

APP=imageProcessing.exe
SRC= utilities.cpp blockMatching.cpp cornerDetection.cpp transformImage.cpp filterImage.cpp loadImages.cpp xml.cpp histogram.cpp sensors.cpp imageProcessing.cpp

ifdef profile
OPT=-pg -g
else
ifdef optim
OPT=-O3
else
OPT=-g
endif
endif

ifdef cuda
CCX=nvcc $(CXXFLAGS) $(OPT)
SRC_CU= blockMatching_cuda.cu 
else
CCX=g++ $(CXXFLAGS) $(OPT)
endif

all: $(APP)

INCLUDE_PATH=/usr/include/libxml2
OPENCV_LIBS=calib3d contrib core features2d flann highgui imgproc legacy ml objdetect ocl photo stitching superres ts video videostab
LIBS=$(OPENCV_LIBS:%=opencv_%) xml2

$(APP): $(SRC_CU:%.cu=%.o) $(SRC:%.cpp=%.o) 
	$(CCX) -o $@ $^ $(LIBS:%=-l%)

%.o: %.cpp
	$(CCX) -c $< $(INCLUDE_PATH:%=-I%)

%.o: %.cu
	$(CCX) -c $< $(INCLUDE_PATH:%=-I%)

profile:
	gprof $(APP) gmon.out > gmon.results

blockMatching_cuda.o:                    	 blockMatching_cuda.cu  
blockMatching_kernel.o:                 	 blockMatching_kernel.cu   
imageProcessing.o: 		 imageProcessing.h   imageProcessing.cpp    globalVars.h globalVars.cpp routines.h
utilities.o:	   		 imageProcessing.h	 utilities.cpp
blockMatching.o:   		 imageProcessing.h	 blockMatching.cpp
cornerDetection.o: 		 imageProcessing.h	 cornerDetection.cpp
transformImage.o:  	     imageProcessing.h	 transformImage.cpp
filterImage.o:	 		 imageProcessing.h	 filterImage.cpp
loadImages.o:	  		 imageProcessing.h	 loadImages.cpp		globalVars.h
xml.o:		   			 imageProcessing.h	 xml.cpp		globalVars.h
histogram.o:	   	 	 imageProcessing.h	 histogram.cpp		globalVars.h
sensors.o:	   			 imageProcessing.h	 sensors.cpp		globalVars.h

run0: $(APP)
	./$(APP) -v images samples

run1: $(APP)
	./$(APP) -v images/screen_12.yuv.png samples

run2: $(APP)
	./$(APP) -sh --irotate=20 --iscale=0.3 --sscale=0.8 --srotate=90 -v images/screen_12.yuv.png samples

run3: $(APP)
	./$(APP) -sh --blockmatching=1 -v images/screen_12.yuv.png samples

run4: $(APP)
	./$(APP) -sh --keypoints -v  images/screen_12.yuv.png samples

run5: $(APP)
	./$(APP) -sh --keypoints -v --ocv images/screen_12.yuv.png samples

run6: $(APP)
	./$(APP) -sh --keypoints --kmeans=4 -v --ocv images/screen_12.yuv.png samples

run7: $(APP)
	./$(APP) -sh  --xml=dec.xml --keypoints --kmeans=10 -v  images/screen_12.yuv.png samples

run8: $(APP)
	 ./$(APP) -v -sh --filter --blockmatching=1 images/screen_10.yuv.png  samples

run9: $(APP)
	 ./$(APP) -v -sh  --keypoints --kmeans=10 --ocv images/screen_8.yuv.png  samples/sample3.png

run10: $(APP)
	 ./$(APP) -v -sh  --blockmatching images/screen_9.yuv.png  samples/sample3.png

run11: $(APP)
	 ./$(APP) -v -sh  --histogram images/screen_9.yuv.png  samples/sample3.png

run12: $(APP)
	 ./$(APP) -v -sh  --histogram  --blockmatching images/screen_9.yuv.png  samples/sample3.png

run13: $(APP)
	 ./$(APP) -v -sh  --sensors images/screen_9.yuv.png

run14: $(APP)
	 ./$(APP) -v -sh  --histogram  --sensors images/flecheref.png 

run15: $(APP)
	 time ./$(APP) -v -sh  --histogram  --blockmatching OrdresDrone/screen_9.yuv.png  OrdresDrone/montes.png
	 
run15cu: $(APP)
	 ./$(APP) -v -sh  --histogram  --blockmatching_cuda OrdresDrone/screen_9.yuv.png  OrdresDrone/montes.png

	 
run16: $(APP)
	 ./$(APP) -v -sh  --histogram  --blockmatching OrdresDrone/screen_16.yuv.png  samples/montes.png

run17: $(APP)
	 ./$(APP) -v -sh  --histogram  --blockmatching OrdresDrone/screen_17.yuv.png  samples/descends.png

run18: $(APP)
	 ./$(APP) -v -sh  --histogram  --blockmatching OrdresDrone/screen_18.yuv.png  samples/droite.png

run19: $(APP)
	 ./$(APP) -v -sh  --histogram  --blockmatching OrdresDrone/screen_19.yuv.png  samples/gauche.png

clean:
	rm -rf *~ $(APP) *.dSYM *.o *.xml
