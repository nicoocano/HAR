#include "imageProcessing.h"
#include "globalVars.h"
#include "routines.h"

#define SENSOR_SIZE 16

class sensor{
  float value;
  int x;
  int y;
  int size;
public:
  sensor(int xi, int yi, int s){
    value = 0.0;
    x = xi;
    y = yi;
    size =s;
  }
  void draw(cv::Mat *image){
    drawRectangle(image,x,y,SENSOR_SIZE,SENSOR_SIZE,value);
  }
  float compute(cv::Mat *image){
    if (!image) return 0.0;
    int im_step = image->step;
    unsigned char *im = (unsigned char*)(image->data);
    
    for (int i=x; i < x+ SENSOR_SIZE; i++){
      for (int j=y;j < y+ SENSOR_SIZE; j++){
	value += im[INDXs(im_step,j,i)];
      }
    }
    value = value /((float) SENSOR_SIZE*SENSOR_SIZE);
    if (Show) draw(image);
    return value;
  }

};

sensor **images_sensors[MAX_IMAGES];

int   createSensors(cv::Mat *image, int id){    
  if (!image ) return 0;
  int nbSensors = 0;
  int im_step = image->step;
  int im_cols = image->cols;
  int im_rows = image->rows;
  for (int i= SENSOR_SIZE; i < im_cols - SENSOR_SIZE; i += 2*SENSOR_SIZE){
    for (int j=SENSOR_SIZE;j < im_rows - SENSOR_SIZE; j += 2*SENSOR_SIZE){
      nbSensors++;
    }
  }

  images_sensors[id] = new sensor *[nbSensors];

  nbSensors =0;
  for (int i= SENSOR_SIZE; i < im_cols - SENSOR_SIZE; i += 2*SENSOR_SIZE){
    for (int j=SENSOR_SIZE;j < im_rows - SENSOR_SIZE; j += 2*SENSOR_SIZE){
      images_sensors[id][nbSensors] = new sensor(i,j,SENSOR_SIZE);
      nbSensors++;
    }
  }
  return nbSensors;
}

void sensors(cv::Mat *image, int id, int nb){    
  
  if (!image ) return;
  unsigned char *im = (unsigned char*)(image->data);
  int im_step = image->step;
  int im_cols = image->cols;
  int im_rows = image->rows;
  for (int i=0;i< nb; i++){
    images_sensors[id][i]->compute(image);
  }
}



void  sensorsImages(){
  for (int i=0; i< nbImages; i++){
    int nb = createSensors(&image_gray[i],i);    
    sensors(&image_gray[i], i, nb);
  }
}


