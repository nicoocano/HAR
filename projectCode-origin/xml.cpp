#include "imageProcessing.h"
#include "routines.h"
#include "globalVars.h"


void generateXML(const char *filename, 
		 cv::Mat &image,
		 std::vector<cv::KeyPoint> &keypoints,
		 std::vector<cv::Point> &boxLeftCorner,
		 std::vector<cv::Point> &boxRightCorner){
  if (!filename) return;
  if (!endWith(".xml",filename)) {
    std::cerr << "generateXML: file name " << filename << " must end with .xml\n";
    return;
  }
  xmlChar tmp[1000];
  int rc;
  xmlTextWriterPtr  writer = xmlNewTextWriterFilename(filename, 0);
  if (writer == NULL) {
    std::cerr << "testXmlwriterFilename: Error creating the xml writer\n";
    return;
  }
  rc = xmlTextWriterStartDocument(writer, NULL, MY_ENCODING, NULL);
  if (rc < 0) {
    std::cerr << "generateXML: Error at xmlTextWriterStartDocument\n";
    return;
  }
  ///////// start keypoints
  for(int k = 0; k < keypoints.size(); k++ ){
    rc = xmlTextWriterStartElement(writer, BAD_CAST "keypoint");    
    if (rc < 0)  std::cerr << "XML generation error \n";
    sprintf((char *) tmp,"%d", (int) keypoints[k].pt.x);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "x",tmp);
    if (rc < 0)  std::cerr << "XML generation error \n";
    sprintf((char *) tmp,"%d", (int) keypoints[k].pt.y);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "y",tmp);
    if (rc < 0)  std::cerr << "XML generation error \n";
    rc = xmlTextWriterEndElement(writer);
    if (rc < 0)  std::cerr << "XML generation error \n";
  }
  ///////// end  keypoints

  ///////// star boxOfKeypoints
  //  rc = xmlTextWriterStartElement(writer, BAD_CAST "boxOfKeypoints");
  //rc = xmlTextWriterEndElement(writer);
  for(int k = 0; k < boxLeftCorner.size(); k++ ){
    rc = xmlTextWriterStartElement(writer, BAD_CAST "boxOfKeypoints");
    if (rc < 0)  std::cerr << "XML generation error \n";
    sprintf((char *) tmp,"%d", (int) boxLeftCorner[k].x);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "xl",tmp);
    if (rc < 0)  std::cerr << "XML generation error \n";
    sprintf((char *) tmp,"%d", (int) boxLeftCorner[k].y);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "yl",tmp);
    if (rc < 0)  std::cerr << "XML generation error \n";

    sprintf((char *) tmp,"%d", (int) boxRightCorner[k].x);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "xr",tmp);
    if (rc < 0)  std::cerr << "XML generation error \n";
    sprintf((char *) tmp,"%d", (int) boxRightCorner[k].y);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "yr",tmp);
    if (rc < 0)  std::cerr << "XML generation error \n";
    rc = xmlTextWriterEndElement(writer);
    if (rc < 0)  std::cerr << "XML generation error \n";
  }
  ///////// end boxOfKeypoints

  rc = xmlTextWriterEndDocument(writer);
  if (rc < 0)  std::cerr << "XML generation error \n";
  xmlFreeTextWriter(writer);
  if (Verbose){
    std::cerr << "Writing XML file: "<< filename << "\n";
  }
}
