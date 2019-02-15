#include "imageProcessing.h"
#include "globalVars.h"

/* 
 * Binarize before matching 
 */

void histogram(unsigned char *im, int im_step, int im_cols, int im_rows, int *histo){
  if (!im || !histo) return;
  for(int i = 0;i < im_rows*im_cols;i++){
      histo[im[i]]++;
  }
}
void otsu(unsigned char *im, int im_step, int im_cols, int im_rows, int *histo){
  //https://en.wikipedia.org/wiki/Otsu%27s_method  
  float sum = 0;
  int total = im_cols*im_rows;
  for (int i = 1; i < MAX_HISTO; ++i)
    sum += i * histo[i];
  float sumB = 0;
  float wB = 0;
  float wF = 0;
  float mB;
  float mF;
  float max = 0.0;
  float between = 0.0;
  float threshold1 = 0.0;
  float threshold2 = 0.0;
  for (int i = 0; i < MAX_HISTO; ++i) {
    wB += histo[i];
    if (wB == 0) continue;
    wF = total - wB;
    if (wF == 0) break;
    sumB += i * histo[i];
    mB = sumB / wB;
    mF = (sum - sumB) / wF;
    between = wB * wF * (mB - mF) * (mB - mF);
    if ( between >= max ) {
      threshold1 = i;
      if ( between > max ) {
	threshold2 = i;
      }
      max = between;            
    }
  }

  int threshold = ( threshold1 + threshold2 ) / 2.0;
  if (Verbose){
    std::cerr << "Otsu threshold found "  <<threshold << "\n";
  }
  
  //binarise l'image  
  for(int i = 0;i < im_rows;i++){
    for(int j = 0;j < im_cols;j++){
      if (im[INDXs(im_step,i,j)] > threshold){
        im[INDXs(im_step,i,j)] = MAX_HISTO-1;
      } else {
        im[INDXs(im_step,i,j)] = 0;
      }
    }
  }
  
}

void histogram(cv::Mat *image,int *histo, int convert){

  if (!image || !histo) return;

  unsigned char *im = (unsigned char*)(image->data);
  int im_step = image->step;
  int im_cols = image->cols;
  int im_rows = image->rows;
  histogram(im,im_step,im_cols,im_rows,histo);
  if (convert) {
    otsu(im,im_step,im_cols,im_rows,histo);
  }

}



void  histogramImages(int convert){
  for (int i=0; i< nbImages; i++){
    for (int j=0;j < MAX_HISTO; j++) image_histograms[i][j] = 0;
    histogram(&image_gray[i], image_histograms[i], convert);
  }
}

void histogramSamples(int convert){
  for (int i=0; i< nbSamples; i++){
    for (int j=0;j < MAX_HISTO; j++) sample_histograms[i][j] = 0;
    histogram(&sample_gray[i],sample_histograms[i], convert);
  }
}
