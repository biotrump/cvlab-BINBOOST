/***************************************************************************
 *   Copyright (C) 2013 by Tomasz Trzcinski                                *
 *   tomasz.trzcinski@epfl.ch                                              *
 ***************************************************************************/

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <bitset>
#include <sys/time.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <omp.h>

using namespace cv;

namespace boostDesc
{
 // gradient assignment type
 enum Assign {
  ASSIGN_HARD=0,
  ASSIGN_BILINEAR=1,
  ASSIGN_SOFT=2,
  ASSIGN_HARD_MAGN=3,
  ASSIGN_SOFT_MAGN=4
 };

 // struct with weak learners' parameters
 struct WeakLearner{
  float thresh;
  int orient;
  int x_min;
  int x_max;
  int y_min;
  int y_max;
  float alpha;
  float beta;

  WeakLearner() :
   thresh (0.0), orient(0), x_min(0), x_max(0), y_min(0), y_max(0), alpha(0.0), beta(0.0)
  {}
 };

 // utility class
 struct Utils
 {
   // compute images of gradients
   static void computeGradientMaps(const Mat& im,
                                   const Assign& gradAssignType,
                                   const int orientQuant,
                                   vector<Mat>& gradMap);

   // compute integral images
   static void computeIntegrals(const vector<Mat>& gradMap,
                                const int orientQuant,
                                vector<Mat>& integralMap);

   // compute the response of a weak learner
   static float computeWLResponse(const WeakLearner& WL,
                                  const int orientQuant,
                                  const vector<Mat>& integralMap);

   // rectify image patch according to the detected interest point
   static void rectifyPatch(const Mat& image,
                            const KeyPoint& kp,
                            const int& patchSize,
                            Mat& patch);

   // get number of miliseconds
   static double getMs();

   // check if the descriptor is binary (by name)
   static bool isBinary(const string& descriptorType);

   // save descriptors to the output file
   static int saveDescriptors(const string& outputFilename,
                              const string& descriptorType,
                              const vector<KeyPoint>& keypoints,
                              const Mat& descriptors);

   // a method to save images matched with the descriptors
   static void saveMatchedImages(const string& img1Filename,
                                 const string& img2Filename,
                                 const vector<KeyPoint>& keypoints1,
                                 const vector<KeyPoint>& keypoints2,
                                 const vector<DMatch>& homography_matches,
                                 const Mat& H,
                                 const string& outputImgFile);

   // faster implementation to match floating-point and binary vectors
   static int matchFloat(const Mat& descriptors1,
                         const vector<Mat>& descriptors2,
                         vector< vector<DMatch> >& matches);
   static int matchFloat(const Mat& descriptors1,
                         const Mat& descriptors2,
                         vector< vector<DMatch> >& matches);
   static int matchHamming(const Mat& descriptors1,
                           const vector<Mat>& descriptors2,
                           vector< vector<DMatch> >& matches);
   static int matchHamming(const Mat& descriptors1,
                           const Mat& descriptors2,
                           vector< vector<DMatch> >& matches);
 };

}
