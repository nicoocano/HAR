#include "imageProcessing.h"
#include "globalVars.h"

/* 
 * Filter before matching 
 */

void filterImage(unsigned char *out, unsigned char *im, int im_step, int im_cols, int im_rows){
  if (!im || !out) return;

  for(int i = 3;i < im_rows-3;i++){
    for(int j = 3;j < im_cols-3;j++){
      double v1 = (2047.0 *(im[INDXs(im_step,i,j+1)] - im[INDXs(im_step,i,j-1)])
		   +913.0 *(im[INDXs(im_step,i,j+2)] - im[INDXs(im_step,i,j-2)])
		   +112.0 *(im[INDXs(im_step,i,j+3)] - im[INDXs(im_step,i,j-3)]))/8418.0;
      //v1 is not in the range NEED FIXING
      out[INDXs(im_step,i,j)] = v1;
    }
  }
}

void filterImage(float *out, float *im, int im_step, int im_cols, int im_rows){
  if (!im || !out) return;

  for(int i = 3;i < im_rows-3;i++){
    for(int j = 3;j < im_cols-3;j++){
      double v1 = (2047.0 *(im[INDXs(im_step,i,j+1)] - im[INDXs(im_step,i,j-1)])
                   +913.0 *(im[INDXs(im_step,i,j+2)] - im[INDXs(im_step,i,j-2)])
                   +112.0 *(im[INDXs(im_step,i,j+3)] - im[INDXs(im_step,i,j-3)]))/8418.0;
      //v1 is not in the range NEED FIXING                                                                                        
      out[INDXs(im_step,i,j)] = v1;
    }
  }
}
 
cv::Mat *filterImage(cv::Mat *image){

  cv::Mat *filtered = NULL;
  if (!image) return filtered;
  //Deep copy of the original                                                                                                                                                                                                         
  filtered = new cv::Mat(image->clone());
  unsigned char *fil = (unsigned char*)(filtered->data);

  unsigned char *im = (unsigned char*)(image->data);
  int im_step = image->step;
  int im_cols = image->cols;
  int im_rows = image->rows;
  filterImage(fil,im,im_step,im_cols,im_rows);
  return filtered;
}


void  filterImages(){
  for (int i=0; i< nbImages; i++){
    image_gray[i] = *filterImage(&image_gray[i]);
  }
}

void filterSamples(){
  for (int i=0; i< nbSamples; i++){
    sample_gray[i] = *filterImage(&sample_gray[i]);
  }
}
