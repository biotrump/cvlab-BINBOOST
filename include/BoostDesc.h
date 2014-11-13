/***************************************************************************
 *   Copyright (C) 2013 by Tomasz Trzcinski                                *
 *   tomasz.trzcinski@epfl.ch                                              *
 ***************************************************************************/

#pragma once

#include <Utils.h>

using namespace cv;

namespace boostDesc
{
 class BGM : public DescriptorExtractor
 {
  public:
  BGM (string _wlFile = "matrices/bgm.bin") :
        wlFile(_wlFile)
   {
    for (unsigned int i=0; i<8; ++i)
     binLookUp[i] = (uchar) 1 << i;
    if (_wlFile.find("txt") != string::npos)
     readWLs();
    else
     readWLsBin();
   };

   ~BGM()
   {
    delete pWLs;
   };

   // interface methods inherited from cv::DescriptorExtractor
   virtual void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
   virtual int descriptorSize() const {return nDim/8;}
   virtual int descriptorType() const {return CV_8UC1;}
   virtual void computeImpl(const Mat&  image, vector<KeyPoint>& keypoints, Mat& descriptors) const
    {compute(image, keypoints, descriptors);}

   virtual void saveWLsBin(const string output) const;

  protected:
   string wlFile;
   int nWLs;
   int nDim;
   WeakLearner* pWLs;
   int orientQuant;
   int patchSize;
   Assign gradAssignType;
   uchar binLookUp[8];

   virtual void readWLs();
   virtual void readWLsBin();
 };

 class LBGM : public BGM
 {
  public:
  LBGM (string _wlFile = "matrices/lbgm.bin")
   {
    wlFile = _wlFile;
    if (wlFile.find("txt") != string::npos)
     readWLs();
    else
     readWLsBin();
   }
   ~LBGM ()
   {
    delete [] betas;
   };

   // interface methods inherited from cv::DescriptorExtractor
   void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
   int descriptorSize() const {return nDim*sizeof(float);}
   int descriptorType() const {return CV_32FC1;}
   void computeImpl(const Mat&  image, vector<KeyPoint>& keypoints, Mat& descriptors) const
    {compute(image, keypoints, descriptors);}

   void saveWLsBin(const string output) const;

  protected:
   float* betas;

   void readWLs();
   void readWLsBin();

 };

 class BinBoost : public BGM
 {
  public:
   BinBoost (string _wlFile = "matrices/binboost.bin") :

   BGM(_wlFile)
   {
    if (_wlFile.find("txt") != string::npos)
     readWLs();
    else
     readWLsBin();
   }
   ~BinBoost ()
   {
     delete [] pWLsArray;
   };

   // interface methods inherited from cv::DescriptorExtractor
   void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
   int descriptorSize() const {return nDim/8;}
   int descriptorType() const {return CV_8UC1;}
   void computeImpl(const Mat&  image, vector<KeyPoint>& keypoints, Mat& descriptors) const
    {compute(image, keypoints, descriptors);}

   void saveWLsBin(const string output) const;

  protected:
   WeakLearner** pWLsArray;

   void readWLs();
   void readWLsBin();
 };


}
