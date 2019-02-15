#ifndef ROUTINES_H
#define ROUTINES_H

//Utilities
void drawRectangle(cv::Mat *image,int coord_j_min, int coord_i_min,int bl_cols, int bl_rows, int val=200);
double wallclock(); //return the wallclock time in seconds              
bool startWith(const char *pre, const char *str);
bool endWith(const char *post, const char *str);

//image processing part
cv::Mat *rotateImage(cv::Mat *image, int theta);
cv::Mat *scaleImage(cv::Mat *image, float scale);
void harrisCornerDetection(cv::Mat *image, std::vector<cv::KeyPoint> &keypoints,float threshold = 0.01);
void filterImage(unsigned char *out, unsigned char *im, int im_step, int im_cols, int im_rows);
cv::Mat *filterImage(cv::Mat *image);
double blockMatching(cv::Mat *image, cv::Mat *block,
     		     int stride, unsigned char *res, int samplenum);

double blockMatchingWithScalingAndRotation(cv::Mat *image, cv::Mat *block,
					   int stride,unsigned char *res, int samplenum);

void filterImage(float *out, float *im, int im_step, int im_cols, int im_rows);


void rotateImages(int theta);
void rotateSamples(int theta);

void scaleImages(float scale);
void scaleSamples(float scale);


void filterImages();
void filterSamples();

void histogramImages(int convert =0);
void histogramSamples(int convert = 0);

void  sensorsImages();

// loading samples
void loadImageSet(int image, const char *dir, int depth); //1 mean it is the image set
int isArgumentADirectory(const char *dir);
void loadOneSampleOrImage(int image, const char *fname);
void showImages();
void showSamples();
void showOneImage(cv::Mat &image);

//XML 

void generateXML(const char *filename,
                 cv::Mat &image,
                 std::vector<cv::KeyPoint> &keypoints,
                 std::vector<cv::Point> &boxLeftCorner,
                 std::vector<cv::Point> &boxRightCorner);

#endif
