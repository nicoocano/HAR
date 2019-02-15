#include "imageProcessing.h"
#include "globalVars.h"
/*
 * Rotate Image
 */
cv::Mat *rotateImage(cv::Mat *image, int theta){
  cv::Mat *rotated = NULL;
  if (!image) return rotated;
  //Deep copy of the original
  rotated = new cv::Mat(image->clone());
  unsigned char *rr = (unsigned char*)(rotated->data);

  unsigned char *im = (unsigned char*)(image->data);
  int im_step = image->step;
  int im_cols = image->cols;
  int im_rows = image->rows;


  int background = 255; // this is the background color - use a suitable value here

  float rads = theta*3.1415926/180.0; // fixed constant PI
  float cs = cos(-rads); // precalculate these values
  float ss = sin(-rads);
  float xcenter = (float)(im_cols)/2.0;   // use float here!
  float ycenter = (float)(im_rows)/2.0;

  for(int r = 0;r < im_rows;r++){
    for(int c = 0;c < im_cols;c++){
      int rorig = ycenter + ((float)(r)-ycenter)*cs - ((float)(c)-xcenter)*ss;
      int corig = xcenter + ((float)(r)-ycenter)*ss + ((float)(c)-xcenter)*cs;
      // now get the pixel value if you can
      int pixel = background; // in case there is no original pixel
      if (rorig >= 0 && rorig < im_rows && corig >= 0 && corig < im_cols) {
	pixel = im[INDXs(im_step,rorig,corig)];
      }
      rr[INDXs(im_step,r,c)] = pixel;
    }
  }

  
  return rotated;
}

void rotateImages(int theta){
  for (int i=0; i< nbImages; i++){
    image_gray[i] = *rotateImage(&image_gray[i],theta);
  }
}

void rotateSamples(int theta){
  for (int i=0; i< nbSamples; i++){
    sample_gray[i] = *rotateImage(&sample_gray[i],theta);
  }
}


/* 
 * Scale Image
 */
cv::Mat *scaleImage(cv::Mat *image, float scale){
  
  if (!image || (scale < 0) || (scale > 1)) return NULL;
  unsigned char *im = (unsigned char*)(image->data);
  int im_step = image->step;
  int im_cols = image->cols;
  int im_rows = image->rows;

  int sc_cols = im_cols*scale;
  int sc_rows = im_rows*scale;
  Mat *scaled_im= new cv::Mat(sc_rows,sc_cols,image->type() /*CV_8UC1*/); 
  unsigned char *sc = (unsigned char*)(scaled_im->data);
  int sc_step = scaled_im->step;

  const float tx = float(im_cols) / sc_cols;
  const float ty = float(im_rows) / sc_rows;

  //printf("Original image %d x %d\n",im_cols, im_rows);
  //printf("Scaled image %d x %d\n",sc_cols, sc_rows);
  //printf("scaling %f:%f\n",tx,ty);

  for (int r = 0; r < sc_rows; r++){
      for (int c = 0; c < sc_cols; c++){
	int x = int(tx * c);
	int y = int(ty * r);
	float dx = tx * c - x;
	float dy = ty * r - y;
	//Stupid version does not do averaging with neighbor pixels
	int pixel = im[INDXs(im_step,y,x)];
	//	printf("%d %d %d --> %d %d\n",y,x,pixel,r,c);
	sc[INDXs(sc_step,r,c)] = pixel;
      }
  }
  return scaled_im;
}


void scaleImages(float scale){
  for (int i=0; i< nbImages; i++){
    image_gray[i] = *scaleImage(&image_gray[i],scale);
  }
}

void scaleSamples(float scale){
  for (int i=0; i< nbSamples; i++){
    sample_gray[i] = *scaleImage(&sample_gray[i],scale);
  }
}

