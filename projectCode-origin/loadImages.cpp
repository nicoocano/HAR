#include "imageProcessing.h"
#include "globalVars.h"

void loadOneSampleOrImage(int image,  const char *fname){
  Mat in_im;
  if (!fname) return;
  // We load the samples given as parameters
  if (fname[0] == '.') return;
  in_im = imread(fname, 1 );
  if (!image){
    if (Verbose)    fprintf(stderr,"Sample loaded %d (%s)\n",nbSamples,fname);
    cvtColor(in_im, sample_gray[nbSamples], CV_BGR2GRAY );
    sample_windows[nbSamples] = strdup(fname);
    nbSamples++;
  } else {
    if (Verbose) fprintf(stderr,"Image loaded %d (%s)\n",nbImages,fname);
    cvtColor(in_im, image_gray[nbImages], CV_BGR2GRAY );
    image_windows[nbImages] = strdup(fname);
    nbImages++;
  }
  return;
}

void loadImageSet(int image,const char *dir, int depth){
  DIR *dp;
  struct dirent *entry;
  struct stat statbuf;
  if((dp = opendir(dir)) == NULL) {
    fprintf(stderr,"cannot open directory: %s\n", dir);
    exit(1);
  }
  chdir(dir);
  while((entry = readdir(dp)) != NULL) {
    lstat(entry->d_name,&statbuf);
    if(S_ISDIR(statbuf.st_mode)) {
      /* Found a directory, but ignore . and .. */
      if(strcmp(".",entry->d_name) == 0 ||
	 strcmp("..",entry->d_name) == 0)
	continue;
      /* Recurse at a new indent level */
      loadImageSet(image,entry->d_name,depth+4);
    } else {
      loadOneSampleOrImage(image,entry->d_name);
    }
  }
  chdir("..");
  closedir(dp);
}


int isArgumentADirectory(const char *dir){
  DIR *dp;
  if (!dir) return false;
  if((dp = opendir(dir)) == NULL) return false;
  return true;
}
