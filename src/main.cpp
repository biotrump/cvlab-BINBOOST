/***************************************************************************
 *   Copyright (C) 2013 by Tomasz Trzcinski                                *
 *   tomasz.trzcinski@epfl.ch                                              *
 ***************************************************************************/

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>

// OpenCv headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/compat.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <BoostDesc.h>
#include <Utils.h>

using namespace std;
using namespace cv;
using namespace boostDesc;

#define VERBOSE 1

// ratio threshold on the nearest neighbor distances (as in SIFT paper, default should be 0.8)
static const float nn_ratio_threshold = 0.8f;

/************************************************************/

struct _CmdLnArgs {

 int mode;             // mode of the program
 string arg1;
 string arg2;
 string descriptor;    // descriptor type

 bool parse(int argc, char *argv[])
 {
  if (argc != 5) return false;

  if (strcmp(argv[1], "--extract") == 0)    { mode = 1; }
  else if (strcmp(argv[1], "--match") == 0) { mode = 2; }
  else { mode = 5; return false; }

  arg1 = (string) argv[2];
  arg2 = (string) argv[3];
  descriptor = (string) argv[4];

  transform(descriptor.begin(), descriptor.end(), descriptor.begin(), ::toupper);

  // check descriptor type
  if (descriptor.find("BGM") == string::npos && descriptor != "LBGM"
      && descriptor.find("BINBOOST") == string::npos && descriptor.find("FPBOOST") == string::npos)
   return false;

  return true;
 }

} CmdLnArgs;

/************************************************************/

int extractDescriptors(const string imgFilename,
                       const string descriptorType,
                       vector<KeyPoint>& keypoints,
                       Mat& descriptors,
                       int verbose = VERBOSE)
{
 // read the image
 Mat image = imread(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
 if (!image.data)
 {
  fprintf(stderr, "[ERROR] Img '%s' cannot be loaded...\n", imgFilename.c_str());
  exit(-2);
 }

 // detect keypoints
 int nrIter = 15;
 Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector (new SurfAdjuster(), 900, 1100, nrIter));
 double start = Utils::getMs();
 detector->detect(image, keypoints);
 if (verbose)
  fprintf(stdout, "[INFO] Img: '%s': %d keypoints detected in %3.2f ms. ",
           imgFilename.c_str(), (int) keypoints.size(), (Utils::getMs() - start)/ (float) nrIter);

 // extract descriptors
 Ptr<DescriptorExtractor> extractor;
 if (descriptorType == "BGM" || descriptorType == "BINBOOST_1" || descriptorType == "BINBOOST_1-256")
  extractor = Ptr<DescriptorExtractor>(new BGM());
 else if (descriptorType == "BGM-HARD" || descriptorType == "BINBOOST_1-HARD")
   extractor = Ptr<DescriptorExtractor>(new BGM("matrices/bgm_hard.bin"));
 else if (descriptorType == "BGM-BILINEAR" || descriptorType == "BINBOOST_1-BILINEAR")
   extractor = Ptr<DescriptorExtractor>(new BGM("matrices/bgm_bilinear.bin"));
 else if (descriptorType == "LBGM" || descriptorType == "FPBOOST" || descriptorType == "FPBOOST_512" || descriptorType == "FPBOOST_512-64")
  extractor = Ptr<DescriptorExtractor>(new LBGM());
 else if (descriptorType == "BINBOOST" || descriptorType == "BINBOOST_128" || descriptorType == "BINBOOST_128-64")
  extractor = Ptr<DescriptorExtractor>(new BinBoost());
 else if (descriptorType == "BINBOOST-128" || descriptorType == "BINBOOST_128-128")
  extractor = Ptr<DescriptorExtractor>(new BinBoost("matrices/binboost_128.bin"));
 else if (descriptorType == "BINBOOST-256" || descriptorType == "BINBOOST_128-256")
  extractor = Ptr<DescriptorExtractor>(new BinBoost("matrices/binboost_256.bin"));

 start = Utils::getMs();
 extractor->compute(image, keypoints, descriptors);
 if (verbose)
  fprintf(stdout, "%d descriptors computed in %3.2f ms.\n",
          (int) descriptors.rows , Utils::getMs() - start);
 return 0;
}

/************************************************************/

int matchImgs(const string img1Filename,
              const string img2Filename,
              const string descriptorType)
{
 // extract keypoints
 vector<KeyPoint> keypoints1, keypoints2;
 Mat descriptors1, descriptors2;
 extractDescriptors(img1Filename, descriptorType, keypoints1, descriptors1);
 extractDescriptors(img2Filename, descriptorType, keypoints2, descriptors2);

 // match descriptors
 vector<vector<DMatch> > knnMatches;
 double start = Utils::getMs();
 if (Utils::isBinary(descriptorType))
  Utils::matchHamming(descriptors1, descriptors2, knnMatches);
 else
  Utils::matchFloat(descriptors1, descriptors2, knnMatches);

#if VERBOSE
 fprintf(stdout, "[INFO] %d descriptors matched against %d descriptors in %3.2f ms.\n",
           (int) descriptors1.rows, descriptors2.rows, Utils::getMs() - start);
#endif

 // sort out the best matches according to Lowe's criterion on the distances' ratio
 vector< DMatch > good_matches;
 for(unsigned int i = 0; i < knnMatches.size(); i++ )
  if (knnMatches[i][0].distance<nn_ratio_threshold*knnMatches[i][1].distance)
   good_matches.push_back( knnMatches[i][0]);

 if (good_matches.size() < 4)
 {
  fprintf(stderr, "[ERROR] Not enough matches found (%d < 4).\n", (int) good_matches.size());
  return 2;
 }

 // get the matching keypoints
 vector<Point2f> match_left, match_right;
 vector<float> distances;
 for(unsigned int i = 0; i < good_matches.size(); i++ )
 {
  match_left.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
  match_right.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
  distances.push_back(good_matches[i].distance);
 }

 Mat correctMatches;
 Mat H = findHomography( match_left, match_right, CV_RANSAC, 3, correctMatches );

 // check if the homography matrix is good enough
 float detMin = 1e-3f;
 if (abs(determinant(H)) < detMin)
 {
  fprintf(stderr, "[ERROR] Bad homography (|det(H)| < %3.3f.\n", detMin);
  return 3;
 }

 // select only the matches that correspond to the found homography
 vector<DMatch> homography_matches;
 for (unsigned int i=0; i < good_matches.size(); ++i)
 {
  if (*correctMatches.ptr<uchar>(i))
   homography_matches.push_back(good_matches[i]);
 }
#if VERBOSE
 fprintf(stdout, "[INFO] Number of matches post-homography: %d (%3.2f%% of the RANSAC input).\n",
         (int) homography_matches.size(), 100.f * (float) homography_matches.size() / (float) good_matches.size());
#endif

 // display matched images
 const string outputImgFile = "matches.png";
 Utils::saveMatchedImages(img1Filename, img2Filename, keypoints1, keypoints2, homography_matches, H, outputImgFile);

 return 0;
}

/************************************************************/

int main(int argc, char** argv)
{
 srand(0);
 initModule_nonfree();

 if (!CmdLnArgs.parse(argc, argv))
 {
  fprintf(stderr, "Usage: %s (--extract <imgFile> <outputFile> <descriptor>\n", argv[0]);
  fprintf(stderr, "\t\t| --match <imgFile1> <imgFile2> <descriptor>)\n");
  return 1;
 }
 
 switch (CmdLnArgs.mode)
 {
  case 1:
  {
   Mat descriptors;
   vector<KeyPoint> keypoints;
   extractDescriptors(CmdLnArgs.arg1, CmdLnArgs.descriptor, keypoints, descriptors);
   Utils::saveDescriptors(CmdLnArgs.arg2, CmdLnArgs.descriptor, keypoints, descriptors);
   break;
  }
  case 2:
  {
   matchImgs(CmdLnArgs.arg1, CmdLnArgs.arg2, CmdLnArgs.descriptor);
   break;
  }
 }

 return 0;
}
