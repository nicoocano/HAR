#include "imageProcessing.h"
#include "globalVars.cpp"
#include "routines.h"

int main(int argc, char *argv[]){
  double tstart, tend;
  ///////////////////////////////////
  // Parse options 
  //////////////////////////////////   
  argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
  option::Stats  stats(usage, argc, argv);
  std::vector<option::Option> options(stats.options_max);
  std::vector<option::Option> buffer(stats.buffer_max);
  option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);
  if (parse.error()) return 1;
  if (options[VERBOSE].count()){
    Verbose = 1;
    std::cerr << "Verbose on"<< "\n";
  }
  if (options[SHOW].count()){
    Show = 1;
    if (Verbose) std::cerr << "Show on"<< "\n";
  }
  if (parse.nonOptionsCount() <1) {
    std::cerr << "Missing image input"<< "\n";
    return 1;
  }
  if (Verbose){ //display option info if verbose mode
    std::cout << "stats.options_max: " << stats.options_max << "\n";
    std::cout << "stats.buffer_max: " << stats.buffer_max << "\n";

    for (option::Option* opt = options[UNKNOWN]; opt; opt = opt->next())
      std::cout << "Unknown option: " << std::string(opt->name,opt->namelen) << "\n";

    for (int i = 0; i < parse.nonOptionsCount(); ++i)
      std::cout << "Non-option #" << i << ": " << parse.nonOption(i) << "\n";
  }
  ///////////////////////////////////
  // load images and samples if given
  ///////////////////////////////////
  const char *si = parse.nonOption(0);
  if (!si) {
    std::cerr << "Missing input image"  << "\n";
    return 1;
  }
  tstart = wallclock();
  if ( isArgumentADirectory(si)){
    loadImageSet(1,si,1);
  } else {
    loadOneSampleOrImage(1,si);
  }
  // load samples if any
  const char *sa = parse.nonOption(1);
  if (sa) {
    if ( isArgumentADirectory(sa)){
      loadImageSet(0,sa,1); // first parameter 0 indicates sample
    } else {
      loadOneSampleOrImage(0,sa);
    }
  }
  tend = wallclock();
  if (Verbose) printf("Loading images tooks %f seconds\n", tend - tstart);
  ////////////////////////////////////
  // appli rotation if asked
  ////////////////////////////////////
  if (options[IROTATION].count() || options[SROTATION].count()){
    tstart = wallclock();
    if (options[IROTATION].count()){
      if (Verbose)  std::cerr << "Image rotation on"  << "\n";
      if (Verbose)   std::cerr << "Rotation angle "  << options[IROTATION].arg << "\n";
      rotateImages(atoi(options[IROTATION].arg));
    }
    if (options[SROTATION].count()){
      if (Verbose)  std::cerr << "Sample rotation on"  << "\n";
      if (Verbose)   std::cerr << "Rotation angle "  << options[SROTATION].arg << "\n";
      rotateSamples(atoi(options[SROTATION].arg));
    }
    tend = wallclock();
    if (Verbose) printf("Rotating images tooks %f seconds\n", tend - tstart);
  }
  ////////////////////////////////////                                                                                                                                 
  // appli scaling if asked   
  ////////////////////////////////////    
  if (options[ISCALING].count()||options[SSCALING].count()){
    tstart = wallclock();
    if (options[ISCALING].count()){
      if (Verbose)  std::cerr << "Image scaling on"  << "\n";
      if (Verbose)   std::cerr << "Scaling factor "  << options[ISCALING].arg << "\n";
      scaleImages(atof(options[ISCALING].arg));
    }
    
    if (options[SSCALING].count()){
      if (Verbose)  std::cerr << "Sample scaling on"  << "\n";
      if (Verbose)   std::cerr << "Scaling factor "  << options[SSCALING].arg << "\n";
      scaleSamples(atof(options[SSCALING].arg));
    }
    tend = wallclock();
    if (Verbose) printf("Scaling images tooks %f seconds\n", tend - tstart);
  }

  //////////////////////////////////// 
  // Do image filtering here
  ////////////////////////////////////         

  if (options[FILTER1].count()){
    if (Verbose)  std::cerr << "Image filtering on"  << "\n";
    tstart = wallclock();
    filterImages();
    filterSamples();
    tend = wallclock();
    if (Verbose) printf("Filtering images tooks %f seconds\n", tend - tstart);
  }

  ////////////////////////////////////                                                                                                                                                                               
  // Compute image and sampkes histogram here                                                                                                                                             
  ////////////////////////////////////

  if (options[HISTOGRAM].count()){
    if (Verbose)  std::cerr << "Image histogram on"  << "\n";
    tstart = wallclock();
    histogramImages(1); //1 to have binarization
    histogramSamples(1); 
    tend = wallclock();
    if (Verbose) printf("Computing histogram tooks %f seconds\n", tend - tstart);
  }


  
  ////////////////////////////////////                                                                                                                                 
  // Do Block Matching here
  ////////////////////////////////////

  if (options[BM].count()){
    tstart = wallclock();
    int coord_j_min = 0;
    int coord_i_min = 0;
    if (Verbose)  std::cerr << "Block matching on"  << "\n";
    int sampleNum=0;
    if (options[BM].arg){
      sampleNum = atoi(options[BM].arg);
      if (sampleNum < 0) sampleNum = 0;
      if (sampleNum >= nbSamples) sampleNum = nbSamples-1 ;
    }
    if (Verbose)  std::cerr << "Using Sample "  << sampleNum <<"\n";
    if (nbImages > 0) {
      double x = blockMatchingWithScalingAndRotation(&(image_gray[0]),&(sample_gray[sampleNum]),1,resultBuffer[sampleNum],sampleNum);
      int *ptj = (int *) &resultBuffer[sampleNum][0];
      int *pti = (int *) &resultBuffer[sampleNum][4];
      coord_j_min = *ptj;
      coord_i_min = *pti;
      if (Show){
	printf("sampleNum = %s, coord_j_min = %d, coord_i_min  = %d\n",sample_windows[sampleNum], coord_j_min,coord_i_min);
	drawRectangle(&image_gray[0],coord_i_min,coord_j_min,sample_gray[sampleNum].cols,sample_gray[sampleNum].rows);
      }
    } else {
      std::cerr << "No images provided for block matching "<<"\n";
    }
    tend = wallclock();
    if (Verbose) printf("Blockmatching images tooks %f seconds\n", tend - tstart);
  }

  ////////////////////////////////////                                                                                                                                 
  // Do KeyPoint here   
  ////////////////////////////////////                     
  
  if (options[KP].count()){
    tstart = wallclock();
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::Point> boxLeftCorner;
    std::vector<cv::Point> boxRightCorner;

    if (!options[OCV].count()){
      if (nbImages){
	harrisCornerDetection(&image_gray[0],keypoints1,0.0);
	if (Verbose){
	  std::cerr << "Found " << keypoints1.size() << " keypoints\n";
	}
	if (Show){
	  cv::drawKeypoints(image_gray[0], keypoints1, image_gray[0]);
	}
      }
      if (nbSamples) {
	harrisCornerDetection(&sample_gray[0],keypoints2,0.0);
	if (Verbose){
	  std::cerr << "Found " << keypoints2.size() << " keypoints\n";
        }
	if (Show){
	  cv::drawKeypoints(sample_gray[0], keypoints2, sample_gray[0]);
	}
      }
    } else {
      OrbFeatureDetector detector1(100),detector2(100);
      // OpenCV version for checking purpose
      // does not work on small images:
      // BRIEF and ORB use a 32x32 patch to get the descriptor.
      // Since it doesn't fit your image, they remove those keypoints (to avoid returning keypoints without descriptor).
      
      if (nbImages) {
	detector1.detect(image_gray[0], keypoints1);
	if (Verbose){
	  std::cerr << "Found " << keypoints1.size() << " keypoints\n";
        }
	if (Show){
	  cv::drawKeypoints(image_gray[0], keypoints1, image_gray[0]);
	}
      }
      if (nbSamples) {
	detector2.detect(sample_gray[0], keypoints2);
	if (Verbose){
	  std::cerr << "Found " << keypoints2.size() << " keypoints\n";
        }
	if (Show && nbSamples){
	  cv::drawKeypoints(sample_gray[0], keypoints2, sample_gray[0]);
	}
      }
    }
    tend = wallclock();
    if (Verbose) printf("Keypoint computation tooks %f seconds\n", tend - tstart);

    if (options[KMEANS].count() && options[KMEANS].arg){
      tstart = wallclock();
      //Lets try kmeans to group keypoints
      int k;
      int clusterCount = atoi(options[KMEANS].arg);  
      Mat labels;  
      int attempts = 3;  
      Mat centers;
      Mat samples(keypoints1.size(), 2, CV_32F);  //x,y
  
      for(k=0; k<keypoints1.size(); k++){
	samples.at<float>(k,0)=keypoints1[k].pt.x;
	samples.at<float>(k,1)=keypoints1[k].pt.y;
      }
      cv::kmeans(samples, 
		 clusterCount, 
		 labels, 
		 TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 1.0), 
		 attempts, 
		 KMEANS_RANDOM_CENTERS, 
		 centers);
  
      //for(k=0; k<keypoints1.size(); k++){
      // int cluster_idx = labels.at<int>(k);
      // printf("Cluster_idx x=%f y=%f --> %d\n",keypoints1[k].pt.x,keypoints1[k].pt.y,cluster_idx);
      // }
  
      // for( k = 0; k < keypoints1.size(); k++ ){
      //int cluster_idx = labels.at<int>(k);
      // circle(image_gray[0],keypoints1[k].pt, 8, colorTab[cluster_idx]);
      // }
      // Need to create the box for each cluster and output it 
      // in an xml descriptor that can then be re-read...
  
      for (int i= 0; i < clusterCount; i++){
	int minXbox = INT_MAX, maxXbox = INT_MIN;
	int minYbox = INT_MAX, maxYbox = INT_MIN;
	int nbkp = 0;
	for( k = 0; k < keypoints1.size(); k++ ){
	  int cluster_idx = labels.at<int>(k);
	  if (cluster_idx ==i){
	    nbkp++;
	    if (keypoints1[k].pt.x >  maxXbox) {
	      maxXbox = keypoints1[k].pt.x;
	    }
	    if (keypoints1[k].pt.x <  minXbox) {
	      minXbox = keypoints1[k].pt.x;
	    }
	
	    if (keypoints1[k].pt.y >  maxYbox) {
	      maxYbox = keypoints1[k].pt.y;
	    }
	    if (keypoints1[k].pt.y <  minYbox) {
	      minYbox = keypoints1[k].pt.y;
	    }
	  }
	}
	if (Verbose) printf("Box (%d,%d) (%d,%d) contains %d keypoints\n",minXbox,minYbox,maxXbox,maxYbox,nbkp);
	boxLeftCorner.push_back(*(new Point(minXbox,minYbox)));
	boxRightCorner.push_back(*(new Point(maxXbox,maxYbox)));
	if (maxXbox != INT_MIN){
	  rectangle(image_gray[0],
		    Point(minXbox,minYbox),
		    Point(maxXbox,maxYbox),
		    colorTab[i]);
	}
      }
      tend = wallclock();
      if (Verbose) printf("Kmeans computation tooks %f seconds\n", tend - tstart);
    }

    if (options[XML].count()){
      tstart = wallclock();
      if (options[XML].arg){      
      // There is a path problem with the XML writer...
	generateXML(options[XML].arg, image_gray[0], keypoints1,
		    boxLeftCorner, boxRightCorner);
      } else {
	std::cerr << "Missing XML file, using :  /tmp/out_imageprocessing.xml\n";
	generateXML("/tmp/out_imageprocessing.xml", image_gray[0], keypoints1,
                    boxLeftCorner, boxRightCorner);
      }
      tend = wallclock();
      if (Verbose) printf("XML computation tooks %f seconds\n", tend - tstart);
    }
  }


  
  //////////////////////////////////// 
  // Check sensors here
  ////////////////////////////////////         

  if (options[SENSORS].count()){
    if (Verbose)  std::cerr << "Sensors on"  << "\n";
    tstart = wallclock();
    sensorsImages();
    tend = wallclock();
    if (Verbose) printf("Sensoring tooks %f seconds\n", tend - tstart);
  }

  ////////////////////////////////////                                                                                                                                 
  // Show images and samples
  ////////////////////////////////////        
  if (Show){ //There is an issue with file names here image and sample may have the same (need a prefix)
    showImages();
    showSamples();  
    waitKey(0);
  }
  return 0;
}




