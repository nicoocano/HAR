#ifndef GLOBAL_VAR_H
#define GLOBAL_VAR_H

int Verbose = 0;
int Show = 0;

unsigned char sampleBuffer[MAX_SAMPLES][SAMPLE_BUFFER_SIZE];
unsigned char imageBuffer[MAX_SAMPLES][IMAGE_BUFFER_SIZE];
unsigned char resultBuffer[MAX_SAMPLES][ANALYSIS_BUFFER_SIZE];

// samples
int nbSamples = 0;
Mat sample_gray[MAX_SAMPLES];
char  *sample_windows[MAX_SAMPLES];
int sample_histograms[MAX_SAMPLES][MAX_HISTO];

// references images
int nbImages = 0;
Mat image_gray[MAX_IMAGES];
char  *image_windows[MAX_IMAGES];
int image_histograms[MAX_IMAGES][MAX_HISTO];


Scalar colorTab[] =
  {
    Scalar(0, 0, 255),
    Scalar(0,255,0),
    Scalar(255,100,100),
    Scalar(255,0,255),
    Scalar(0,255,255)
  };

enum OptionType
  {
    UNUSED = 0, DISABLED = 1, ENABLED = 2
  };


enum  optionIndex { UNKNOWN, HELP, OCV, KP, DESC, BM, VERBOSE, IROTATION, ISCALING, SROTATION, SSCALING, SHOW, KMEANS, XML, FILTER1, HISTOGRAM,SENSORS};

const option::Descriptor usage[] =
  {
    {UNKNOWN, ENABLED,"" , ""    ,option::Arg::None, "USAGE: example [options]\n\n" 
                                                     "Options:" },
    {HELP,    ENABLED,"" , "help",option::Arg::None, "  --help  \tPrint usage and exit." },
    {OCV,     ENABLED, "", "ocv",option::Arg::None, "  --ocv, -o  \tuse opencv for keypoints." },
    {UNKNOWN, ENABLED,"" ,  ""   ,option::Arg::None, "\nExamples:\n" 
                                               "  example --unknown -- --this_is_no_option\n"
                                               "  example -unk --plus -ppp file1 file2\n" },

    {KP,    ENABLED,"" , "keypoints",option::Arg::None, "  --keypoints \tcompute keypoints." },

    {KMEANS,    ENABLED,"" , "kmeans",option::Arg::Optional, "  --kmeans \tclusterize." },

    {DESC,    ENABLED,"d" , "desc",option::Arg::None, "  --desc  \tProduce keypoint descriptor." },

    {BM,    ENABLED,"" , "blockmatching",option::Arg::Optional, "  --blockmatching  \tCompute block matching." },

    {VERBOSE,    ENABLED,"v" , "verbose",option::Arg::None, "  --verbose  \tVerbose mode." },

    {IROTATION,   ENABLED,"" , "irotate",option::Arg::Optional, "  --irotate  \tperform rotation first on input image(s)." },
    {SROTATION,   ENABLED,"" , "srotate",option::Arg::Optional, "  --srotate  \tperform rotation first on sample image(s)." },

    {ISCALING,    ENABLED,"" , "iscale",option::Arg::Optional, "  --iscale  \tperform scaling first on input image(s)." },
    {SSCALING,    ENABLED,"" , "sscale",option::Arg::Optional, "  --sscale  \tperform scaling first on sample image(s)." },

    {SHOW,    ENABLED,"s" , "show",option::Arg::None, "  --show  \tshow images or samples in a window(s)." },

    {XML,    ENABLED,"" , "xml",option::Arg::Optional, "  --xml  \toutput descriptor in XML file given as argument." },

    {FILTER1,    ENABLED,"" , "filter",option::Arg::Optional, "  --filter  \tfilter image." },

    {HISTOGRAM,    ENABLED,"" , "histogram",option::Arg::Optional, "  --histogram  \tcompute an histogram of the image." },

    {SENSORS,    ENABLED,"" , "sensors",option::Arg::None, "  --sensors  \tcompute sensors value on the image (do a binarizarion before)." },

    {0,0,0,0,0,0}
};

#endif
