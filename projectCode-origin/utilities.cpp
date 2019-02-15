#include "imageProcessing.h"
#include "globalVars.h"


bool startWith(const char *pre, const char *str){
  if (!pre || !str) return false;
  return strncmp(pre, str, strlen(pre)) == 0;
}

bool endWith(const char *post, const char *str){
  if (!post || !str) return false;
  if (strlen(post) > strlen(str)) return false;
  return strncmp(post, str+(strlen(str)-strlen(post)),strlen(post)) == 0;
}

void showImages(){
  for (int i=0; i< nbImages; i++){
    namedWindow(image_windows[i], CV_WINDOW_AUTOSIZE );
    imshow(image_windows[i],image_gray[i]);
  }
}

void showSamples(){
  for (int i=0; i< nbSamples; i++){
    namedWindow(sample_windows[i], CV_WINDOW_AUTOSIZE );
    imshow(sample_windows[i],sample_gray[i]);
  }
}

void showOneImage(cv::Mat &image){
  namedWindow("tempView", CV_WINDOW_AUTOSIZE );
  imshow("tempView",image);
}

void drawRectangle(cv::Mat *image,int coord_j_min, int coord_i_min,int bl_cols, int bl_rows, int val){
  if (!image) return;

  Point matchLoc1(coord_j_min, coord_i_min);
  rectangle(*image, matchLoc1, Point( matchLoc1.x + bl_cols, matchLoc1.y + bl_rows), Scalar::all(val), 2, 8, 0 );

}

/*
 * Display normalize float rep
 */

void displayNormalize(cv::Mat *image,float *imf){
  
  if (!imf|| !image) return;
  int im_step = image->step;
  int im_cols = image->cols; //x, width                                                                                                                                                        
  int im_rows = image->rows; //y, height      

  cv::Mat *tmp = new cv::Mat(image->clone());
  unsigned char *tm = (unsigned char*)(tmp->data);
  for (int y = 0; y < im_rows ; y++){
    for (int x = 0; x < im_cols; x++){
      int vv = 255.0f*imf[INDXs(im_step,y,x)];
      //      printf("vv=%d (%f)\n",vv,imf[INDXs(im_step,y,x)]);
      tm[INDXs(im_step,y,x)] = vv;
    }
  }

  namedWindow( "displayNormalize", CV_WINDOW_AUTOSIZE );
  imshow( "displayNormalize",*tmp);
}

//return the wallclock time in seconds                                                                                                                                                                                                   
double wallclock(){
  struct timeval tv;
  double t;

  gettimeofday(&tv, NULL);

  t = (double)tv.tv_sec;
  t += ((double)tv.tv_usec)/1000000.0;
  return t;
}
