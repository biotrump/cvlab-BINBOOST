/***************************************************************************
 *   Copyright (C) 2013 by Tomasz Trzcinski                                *
 *   tomasz.trzcinski@epfl.ch                                              *
 ***************************************************************************/

#include <BoostDesc.h>

namespace boostDesc
{

void BGM::readWLs()
{
 FILE* fid = fopen(wlFile.c_str(),"rt");
 if (!fid)
 {
  fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
  exit(-3);
 }
 fscanf(fid,"%d\n",&nWLs);
 fscanf(fid,"%d\n", &orientQuant);
 fscanf(fid,"%d\n", &patchSize);
 int iGradAssignType;
 fscanf(fid,"%d\n", &iGradAssignType);
 gradAssignType = (Assign) iGradAssignType;
 pWLs = new WeakLearner[nWLs];
 nDim = nWLs;
 for (int i = 0;i < nWLs;i++)
 {
  fscanf(fid,"%f %d %d %d %d %d %f\n",
         &pWLs[i].thresh,
         &pWLs[i].orient,
         &pWLs[i].y_min,
         &pWLs[i].y_max,
         &pWLs[i].x_min,
         &pWLs[i].x_max,
         &pWLs[i].alpha);
 }
 fclose(fid);
}

/**********************************************************************************************/

void BGM::readWLsBin()
{
 FILE* fid = fopen(wlFile.c_str(),"rb");
 if (!fid)
 {
  fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
  exit(-3);
 }
 fread(&nWLs,sizeof(int), 1,fid);
 fread(&orientQuant,sizeof(int), 1,fid);
 fread(&patchSize,sizeof(int), 1,fid);
 int iGradAssignType;
 fread(&iGradAssignType,sizeof(int), 1,fid);
 gradAssignType = (Assign) iGradAssignType;
 pWLs = new WeakLearner[nWLs];
 nDim = nWLs;
 for (int i = 0;i < nWLs;i++)
 {
  fread(&pWLs[i].thresh,sizeof(float), 1,fid);
  fread(&pWLs[i].orient,sizeof(int), 1,fid);
  fread(&pWLs[i].y_min,sizeof(int), 1,fid);
  fread(&pWLs[i].y_max,sizeof(int), 1,fid);
  fread(&pWLs[i].x_min,sizeof(int), 1,fid);
  fread(&pWLs[i].x_max,sizeof(int), 1,fid);
  fread(&pWLs[i].alpha,sizeof(float), 1,fid);
 }
 fclose(fid);
}

/**********************************************************************************************/

void BGM::saveWLsBin(const string output) const
{
 FILE* fid = fopen(output.c_str(),"wb");
 if (!fid)
 {
  fprintf(stderr, "[ERROR] Cannot save weak learners to '%s'.\n", output.c_str());
  exit(-3);
 }
 fwrite(&nWLs,sizeof(int), 1,fid);
 fwrite(&orientQuant,sizeof(int), 1,fid);
 fwrite(&patchSize,sizeof(int), 1,fid);
 fwrite(&gradAssignType,sizeof(int), 1,fid);
 for (int i = 0;i < nWLs;i++)
 {
  fwrite(&pWLs[i].thresh,sizeof(float), 1,fid);
  fwrite(&pWLs[i].orient,sizeof(int), 1,fid);
  fwrite(&pWLs[i].y_min,sizeof(int), 1,fid);
  fwrite(&pWLs[i].y_max,sizeof(int), 1,fid);
  fwrite(&pWLs[i].x_min,sizeof(int), 1,fid);
  fwrite(&pWLs[i].x_max,sizeof(int), 1,fid);
  fwrite(&pWLs[i].alpha,sizeof(float), 1,fid);
 }
 fclose(fid);
}

/**********************************************************************************************/

void BGM::compute( const Mat& image, vector<KeyPoint>& keypoints,
                   Mat& descriptors ) const
{
 // initialize the variables
 descriptors = Mat::zeros(keypoints.size(), ceil(nDim/8), descriptorType());
 vector<Mat> gradMap, integralMap;

 // iterate through all the keypoints
 for (unsigned int i=0; i<keypoints.size(); ++i)
 {
  // rectify the patch around a given keypoint
  Mat patch;
  Utils::rectifyPatch(image, keypoints[i], patchSize, patch);

  // compute gradient maps (and integral gradient maps)
  Utils::computeGradientMaps(patch, gradAssignType, orientQuant, gradMap);
  Utils::computeIntegrals(gradMap, orientQuant, integralMap);

  // compute the responses of the weak learners
  uchar* desc = descriptors.ptr<uchar>(i);
  for (int j = 0;j < nWLs;++j)
   desc[j/8] |= (Utils::computeWLResponse(pWLs[j], orientQuant, integralMap) >= 0) ? binLookUp[j%8] : 0;

  // clean-up
  patch.release();
  gradMap.clear();
  integralMap.clear();
 }
}

/**********************************************************************************************/

void LBGM::readWLs()
{
 FILE* fid = fopen(wlFile.c_str(),"rt");
 fscanf(fid,"%d\n",&nWLs);
 fscanf(fid,"%d\n", &orientQuant);
 fscanf(fid,"%d\n", &patchSize);
 int iGradAssignType;
 fscanf(fid,"%d\n", &iGradAssignType);
 gradAssignType = (Assign) iGradAssignType;
 pWLs = new WeakLearner[nWLs];
 for (int i = 0;i < nWLs;i++)
 {
  fscanf(fid,"%f %d %d %d %d %d %f\n",
         &pWLs[i].thresh,
         &pWLs[i].orient,
         &pWLs[i].y_min,
         &pWLs[i].y_max,
         &pWLs[i].x_min,
         &pWLs[i].x_max,
         &pWLs[i].alpha);
 }
 fscanf(fid, "%d\n", &nDim);
 betas = new float[nDim*nWLs];
 for (int i = 0;i < nWLs;i++)
 {
  for (int d=0; d<nDim; ++d)
   fscanf(fid, "%f ", &betas[i*nDim+d]);
  fscanf(fid, "\n");
 }
 fclose(fid);
}

/**********************************************************************************************/

void LBGM::readWLsBin()
{
 FILE* fid = fopen(wlFile.c_str(),"rb");
 if (!fid)
 {
  fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
  exit(-3);
 }
 fread(&nDim,sizeof(int), 1,fid);
 fread(&nWLs,sizeof(int), 1,fid);
 fread(&orientQuant,sizeof(int), 1,fid);
 fread(&patchSize,sizeof(int), 1,fid);
 int iGradAssignType;
 fread(&iGradAssignType,sizeof(int), 1,fid);
 gradAssignType = (Assign) iGradAssignType;
 pWLs = new WeakLearner[nWLs];
 for (int i = 0;i < nWLs;i++)
 {
  fread(&pWLs[i].thresh,sizeof(float), 1,fid);
  fread(&pWLs[i].orient,sizeof(int), 1,fid);
  fread(&pWLs[i].y_min,sizeof(int), 1,fid);
  fread(&pWLs[i].y_max,sizeof(int), 1,fid);
  fread(&pWLs[i].x_min,sizeof(int), 1,fid);
  fread(&pWLs[i].x_max,sizeof(int), 1,fid);
  fread(&pWLs[i].alpha,sizeof(float), 1,fid);
 }
 betas = new float[nDim*nWLs];
 for (int i = 0;i < nWLs;i++)
  for (int d=0; d<nDim; ++d)
   fread(&betas[i*nDim+d], sizeof(float), 1, fid);
 fclose(fid);
}

/**********************************************************************************************/

void LBGM::saveWLsBin(const string output) const
{
 FILE* fid = fopen(output.c_str(),"wb");
 if (!fid)
 {
  fprintf(stderr, "[ERROR] Cannot save weak learners to '%s'.\n", wlFile.c_str());
  exit(-3);
 }
 fwrite(&nDim,sizeof(int), 1,fid);
 fwrite(&nWLs,sizeof(int), 1,fid);
 fwrite(&orientQuant,sizeof(int), 1,fid);
 fwrite(&patchSize,sizeof(int), 1,fid);
 fwrite(&gradAssignType,sizeof(int), 1,fid);
 for (int i = 0;i < nWLs;i++)
 {
  fwrite(&pWLs[i].thresh,sizeof(float), 1,fid);
  fwrite(&pWLs[i].orient,sizeof(int), 1,fid);
  fwrite(&pWLs[i].y_min,sizeof(int), 1,fid);
  fwrite(&pWLs[i].y_max,sizeof(int), 1,fid);
  fwrite(&pWLs[i].x_min,sizeof(int), 1,fid);
  fwrite(&pWLs[i].x_max,sizeof(int), 1,fid);
  fwrite(&pWLs[i].alpha,sizeof(float), 1,fid);
 }
 for (int i = 0;i < nWLs;i++)
  for (int d=0; d<nDim; ++d)
   fwrite(&betas[i*nDim+d], sizeof(float), 1, fid);
 fclose(fid);
}

/**********************************************************************************************/

void LBGM::compute( const Mat& image, vector<KeyPoint>& keypoints,
                   Mat& descriptors ) const
{
 // initialize the variables
 descriptors = Mat::zeros(keypoints.size(), nDim, descriptorType());
 vector<Mat> gradMap, integralMap;

 // iterate through all the keypoints
 for (unsigned int i=0; i<keypoints.size(); ++i)
 {
  // rectify the patch around a given keypoint
  Mat patch;
  Utils::rectifyPatch(image, keypoints[i], patchSize, patch);

  // compute gradient maps (and integral gradient maps)
  Utils::computeGradientMaps(patch, gradAssignType, orientQuant, gradMap);
  Utils::computeIntegrals(gradMap, orientQuant, integralMap);

  // compute the responses of the weak learners
  std::bitset<512> wlResponses;
  for (int j = 0;j < nWLs;++j)
   wlResponses[j] = (Utils::computeWLResponse(pWLs[j], orientQuant, integralMap) >= 0) ? 1 : 0;

  float* desc = descriptors.ptr<float>(i);
  for (int d=0; d<nDim; ++d)
  {
   for (int wl=0; wl<nWLs; ++wl)
    desc[d] += (wlResponses[wl] ) ? betas[wl*nDim+d] : -betas[wl*nDim+d];
  }

  // clean-up
  patch.release();
  gradMap.clear();
  integralMap.clear();
 }
}

/**********************************************************************************************/

void BinBoost::readWLs()
{
 FILE* fid = fopen(wlFile.c_str(),"rt");
 fscanf(fid, "%d\n", &nDim);
 fscanf(fid,"%d\n", &orientQuant);
 fscanf(fid,"%d\n", &patchSize);
 int iGradAssignType;
 fscanf(fid,"%d\n", &iGradAssignType);
 gradAssignType = (Assign) iGradAssignType;
 pWLsArray = new WeakLearner*[nDim];
 for (int d = 0; d<nDim; ++d)
 {
  fscanf(fid,"%d\n",&nWLs);
  pWLsArray[d] = new WeakLearner[nWLs];
  for (int i = 0;i < nWLs;i++)
  {
   fscanf(fid,"%f %d %d %d %d %d %f %f\n",
          &pWLsArray[d][i].thresh,
          &pWLsArray[d][i].orient,
          &pWLsArray[d][i].y_min,
          &pWLsArray[d][i].y_max,
          &pWLsArray[d][i].x_min,
          &pWLsArray[d][i].x_max,
          &pWLsArray[d][i].alpha,
          &pWLsArray[d][i].beta);
  }
 }

 fclose(fid);
}

/**********************************************************************************************/

void BinBoost::readWLsBin()
{
 FILE* fid = fopen(wlFile.c_str(),"rb");
 if (!fid)
 {
  fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
  exit(-3);
 }
 fread(&nDim,sizeof(int), 1,fid);
 fread(&orientQuant,sizeof(int), 1,fid);
 fread(&patchSize,sizeof(int), 1,fid);
 int iGradAssignType;
 fread(&iGradAssignType,sizeof(int), 1,fid);
 gradAssignType = (Assign) iGradAssignType;
 pWLsArray = new WeakLearner*[nDim];
 for (int d = 0; d<nDim; ++d)
 {
  fread(&nWLs,sizeof(int), 1,fid);
  pWLsArray[d] = new WeakLearner[nWLs];
  for (int i = 0;i < nWLs;i++)
  {
   fread(&pWLsArray[d][i].thresh,sizeof(float), 1,fid);
   fread(&pWLsArray[d][i].orient,sizeof(int), 1,fid);
   fread(&pWLsArray[d][i].y_min,sizeof(int), 1,fid);
   fread(&pWLsArray[d][i].y_max,sizeof(int), 1,fid);
   fread(&pWLsArray[d][i].x_min,sizeof(int), 1,fid);
   fread(&pWLsArray[d][i].x_max,sizeof(int), 1,fid);
   fread(&pWLsArray[d][i].alpha,sizeof(float), 1,fid);
   fread(&pWLsArray[d][i].beta,sizeof(float), 1,fid);
  }
 }
 fclose(fid);
}

/**********************************************************************************************/

void BinBoost::saveWLsBin(const string output) const
{
 FILE* fid = fopen(output.c_str(),"wb");
 if (!fid)
 {
  fprintf(stderr, "[ERROR] Cannot save weak learners to '%s'.\n", output.c_str());
  exit(-3);
 }
 fwrite(&nDim,sizeof(int), 1,fid);
 fwrite(&orientQuant,sizeof(int), 1,fid);
 fwrite(&patchSize,sizeof(int), 1,fid);
 fwrite(&gradAssignType,sizeof(int), 1,fid);
 for (int d = 0; d<nDim; ++d)
 {
  fwrite(&nWLs,sizeof(int), 1,fid);
  for (int i = 0;i < nWLs;i++)
  {
   fwrite(&pWLsArray[d][i].thresh,sizeof(float), 1,fid);
   fwrite(&pWLsArray[d][i].orient,sizeof(int), 1,fid);
   fwrite(&pWLsArray[d][i].y_min,sizeof(int), 1,fid);
   fwrite(&pWLsArray[d][i].y_max,sizeof(int), 1,fid);
   fwrite(&pWLsArray[d][i].x_min,sizeof(int), 1,fid);
   fwrite(&pWLsArray[d][i].x_max,sizeof(int), 1,fid);
   fwrite(&pWLsArray[d][i].alpha,sizeof(float), 1,fid);
   fwrite(&pWLsArray[d][i].beta,sizeof(float), 1,fid);
  }
 }
 fclose(fid);
}

/**********************************************************************************************/

void BinBoost::compute( const Mat& image, vector<KeyPoint>& keypoints,
                        Mat& descriptors ) const
{
 // initialize the variables
 descriptors = Mat::zeros(keypoints.size(), ceil(nDim/8), descriptorType());
 vector<Mat> gradMap, integralMap;

 // iterate through all the keypoints
 for (unsigned int i=0; i<keypoints.size(); ++i)
 {
  // rectify the patch around a given keypoint
  Mat patch;
  Utils::rectifyPatch(image, keypoints[i], patchSize, patch);

  // compute gradient maps (and integral gradient maps)
  Utils::computeGradientMaps(patch, gradAssignType, orientQuant, gradMap);
  Utils::computeIntegrals(gradMap, orientQuant, integralMap);

  // compute the responses of the weak learners
  float resp;
  for (int d = 0; d < nDim; ++d)
  {
   uchar* desc = descriptors.ptr<uchar>(i);
   resp = 0;
   for (int wl = 0;wl < nWLs; wl++)
    resp += (Utils::computeWLResponse(pWLsArray[d][wl], orientQuant, integralMap) >= 0) ? pWLsArray[d][wl].beta : -pWLsArray[d][wl].beta;
   desc[d/8] |= (resp >= 0) ? binLookUp[d%8] : 0;
  }

  // clean-up
  patch.release();
  gradMap.clear();
  integralMap.clear();
 }
}

}
