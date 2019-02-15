#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <sys/socket.h>
#include <dirent.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <limits.h>
#include <opencv2/opencv.hpp>
#include <libxml/parser.h>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#include "optionparser-1.3/src/optionparser.h"


#define INDXr(i,j)  ((step) * (i) + (j*3) + 2)
#define INDXg(i,j)  ((step) * (i) + (j*3) + 1)
#define INDXb(i,j)  ((step) * (i) + (j*3) + 0)
#define INDX(i,j)   ((step) * (i) + (j) + 0)
#define INDXs(s,i,j)   ((s) * (i) + (j) + 0)
#define ABS(a) ((a)<0 ? -(a) : (a))

#define DATA_OFFSET 16
#define IMAGE_BUFFER_SIZE 400000
#define SAMPLE_BUFFER_SIZE 100000
#define ANALYSIS_BUFFER_SIZE 1000
#define PROTOCOLE_BUFFER_SIZE 1000
#define CHUCK_BUFFER_SIZE 14000
#define MAX_SAMPLES 100
#define MAX_IMAGES 100
#define MAX_HISTO 256


#define MY_ENCODING "UTF-8"

using namespace cv;
#endif
