#include "imageProcessing.h"
#include "globalVars.h"

/*
 * Harris corner detection
 */
void harrisCornerDetection(cv::Mat *image,  std::vector<cv::KeyPoint> &keypoints,  float threshold){
  
  if (!image) return;

  std::vector<float> keypointscore;

  unsigned char *im = (unsigned char*)(image->data);
  int im_step = image->step;
  int im_cols = image->cols; //x, width
  int im_rows = image->rows; //y, height

  float *diffx = new float[im_cols*im_rows];
  float *diffy = new float[im_cols*im_rows];
  float *diffxy = new float[im_cols*im_rows];

  float *imf= new float[im_cols*im_rows];
  float minx = FLT_MAX , maxx = FLT_MIN;
  for (int y = 0; y < im_rows ; y++){
    for (int x = 0; x < im_cols; x++){
      if (im[INDXs(im_step,y,x)] > maxx) maxx = im[INDXs(im_step,y,x)];
      if (im[INDXs(im_step,y,x)] < minx) minx = im[INDXs(im_step,y,x)];
    }
  }

  //printf("Min = %f, Max = %f\n",minx,maxx);
  for (int y = 0; y < im_rows ; y++){
    for (int x = 0; x < im_cols; x++){
      imf[INDXs(im_step,y,x)]  =  (im[INDXs(im_step,y,x)] - minx) / (maxx-minx);
      //      printf("%f\n",imf[INDXs(im_step,y,x)]);
    }
  }

  // for each line
  int u = 1;
  int y,x;
  for (y = u; y < im_rows - u; y++){
    for (x = u; x < im_cols - u; x++){
      float h = ((imf[INDXs(im_step,y-u,x+u)] + imf[INDXs(im_step,y+0,x+u)] +imf[INDXs(im_step,y+u,x+u)])-
                 (imf[INDXs(im_step,y-u,x-u)] + imf[INDXs(im_step,y+0,x-u)] + imf[INDXs(im_step,y+u,x-u)]));// * 0.166666667f;
      float v = ((imf[INDXs(im_step,y+u,x-u)] + imf[INDXs(im_step,y+u,x+0)] +imf[INDXs(im_step,y+u,x+u)])-
                 (imf[INDXs(im_step,y-u,x-u)] + imf[INDXs(im_step,y-u,x+0)] + imf[INDXs(im_step,y-u,x+u)]));// * 0.166666667f;
      //printf("%f %f\n",h,v);
      diffx[INDXs(im_step,y,x)] = h * h  *2.0;
      diffy[INDXs(im_step,y,x)] = v * v  *2.0; // seems better with scaling
      diffxy[INDXs(im_step,y,x)] = h * v *2.0; 
      if (diffxy[INDXs(im_step,y,x)] < 0) diffxy[INDXs(im_step,y,x)] = - diffxy[INDXs(im_step,y,x)];
    }
  }
  //displayNormalize(image,diffx);
  // need a convolution filtre orthogonal au gradient
  float *odiffx = new float[im_cols*im_rows];
  float *odiffy = new float[im_cols*im_rows];
  float *odiffxy = new float[im_cols*im_rows];
  for (y = 1; y < im_rows - 1; y++){
    for (x = 1; x < im_cols - 1; x++){
      odiffx[INDXs(im_step,y,x)] = (diffx[INDXs(im_step,y-1,x)]
				    +diffx[INDXs(im_step,y,x)]
				    +diffx[INDXs(im_step,y+1,x)])*0.3;
      odiffy[INDXs(im_step,y,x)] = (diffy[INDXs(im_step,y,x-1)]+diffy[INDXs(im_step,y,x)]+diffy[INDXs(im_step,y,x+1)])*0.33;
      odiffxy[INDXs(im_step,y,x)] = (diffxy[INDXs(im_step,y,x-1)]+diffxy[INDXs(im_step,y,x)]+diffxy[INDXs(im_step,y,x+1)])*0.33;
    }
  }
  delete diffx;
  delete diffy;
  delete diffxy;

  //filterImage(odiffx,diffx, im_step, im_cols, im_rows);
  //displayNormalize(image,odiffx); 

  float *kpmap = new float[im_cols*im_rows];
  for (int y = 0; y < im_rows; y++){
    for (int x = 0; x < im_cols; x++){
      kpmap[INDXs(im_step,y,x)] = FLT_MIN;
    }
  }
  
  // Compute Harris corner response
  float k = 0.2; //Sensitivity factor (0<k<0.25)
  float M1, M2;
  for (int y = 1; y < im_rows - 1; y++){
    for (int x = 1; x < im_cols - 1; x++){
      float A = odiffx[INDXs(im_step,y,x)];
      float B = odiffy[INDXs(im_step,y,x)];
      float C = odiffxy[INDXs(im_step,y,x)];

      // Les valeurs de M sont positives au voisinage d’un coin, 
      // négatives au voisinage d’un contour 
      // et faibles dans une région d’intensité constante.
      float det = ((A * B) - (C * C));
      float t = (A-B)+(A-B) +4.0*(C*C);
      if (t < 0) t = -t;
      t = sqrt(t);
      float vp1 = 0.5*(A+B +t);
      float vp2 = 0.5*(A+B -t);
      M2 = vp1*vp2 -k*(vp1+vp2);
      M1 = det - (k * ((A + B) * (A + B))); //does not wok with normalized values????
      // printf("M1=%f M2=%f M1-M2=%f\n",M1,M2,M1-M2);

      if (M2 > threshold){
       	//printf("Point of interest in %d %d  (%f)\n",y,x,M2);
	//pi[INDXs(im_step,y,x) 
	//should be computing a descriptor for each corner...

	int noNeigborgh = 1;
	int bs = 8;
	int xm =0, ym =0;
	for (int ly = MAX(0,y-bs); ly < MIN(im_rows,y+bs); ly++){
	  for (int lx = MAX(0,x-bs) ;lx < MIN(im_cols,x+bs); lx++){
	    if ((FLT_MIN < kpmap[INDXs(im_step,ly,lx)]) && (kpmap[INDXs(im_step,ly,lx)] > M2)){
	      xm = lx; ym = ly;
	      kpmap[INDXs(im_step,ly,lx)] = FLT_MIN;
	      break;
	    }
	  }	
	}
	if (xm == 0.0) {
	  //printf("Point of interest in %d %d  (%f)\n",y,x,M2); 
	  kpmap[INDXs(im_step,y,x)] = M2;
 	}
      }
    }
  }
  
  for (int y = 1; y < im_rows - 1; y++){
    for (int x = 1; x < im_cols - 1; x++){
      if (kpmap[INDXs(im_step,y,x)] > FLT_MIN){
	keypoints.push_back(*(new cv::KeyPoint(x,y,4)));
	keypointscore.push_back(kpmap[INDXs(im_step,y,x)]);
      }
    }
  }
  delete kpmap;
  delete odiffx;
  delete odiffy;
  delete odiffxy;  
}
