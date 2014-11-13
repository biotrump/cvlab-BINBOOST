/***************************************************************************
 *   Copyright (C) 2013 by Tomasz Trzcinski                                *
 *   tomasz.trzcinski@epfl.ch                                              *
 ***************************************************************************/

#include <Utils.h>

namespace boostDesc
{

void Utils::computeGradientMaps(const Mat& im,
                                const Assign& gradAssignType,
                                const int orientQuant,
                                vector<Mat>& gradMap)
{
 Mat derivx(im.size(), CV_32FC1);
 Mat derivy(im.size(), CV_32FC1);

 Sobel(im, derivx, derivx.depth(), 1, 0);
 Sobel(im, derivy, derivy.depth(), 0, 1);

 for (int i=0; i<orientQuant; ++i)
  gradMap.push_back(Mat::zeros(im.size(), CV_8UC1));

 //Fill in temp matrices with respones to edge detection
 double binSize = (2*M_PI) / orientQuant;
 int index, index2;
 double binCenter, weight;
 const float* pDerivx = derivx.ptr<float>();
 const float* pDerivy = derivy.ptr<float>();
 for (int i=0; i<im.rows; ++i)
 {
  for (int j=0; j<im.cols; ++j)
  {
   float gradMagnitude = sqrt((*pDerivx)*(*pDerivx)+(*pDerivy)*(*pDerivy));
   if ( gradMagnitude > 20)
   {
    double theta = atan2(*pDerivy, *pDerivx);
    theta = (theta<0) ? theta+2*M_PI : theta;
    index = int(theta/binSize);
    index = (index==orientQuant) ? 0 : index;

    switch (gradAssignType)
    {
     case ASSIGN_HARD:
      gradMap[index].at<uchar>(i,j) = 1;
      break;
     case ASSIGN_HARD_MAGN:
      gradMap[index].at<uchar>(i,j) = round(gradMagnitude);
      break;
     case ASSIGN_BILINEAR:
      index2=ceil(theta/binSize);
      index2=(index2==orientQuant) ? 0 : index2;
      binCenter  = (index+.5)*binSize;
      weight = 1 - abs(theta-binCenter)/binSize;
      gradMap[index].at<uchar>(i,j) = (int) round(255*weight);
      gradMap[index2].at<uchar>(i,j) = (int) round(255*(1-weight));
      break;
     case ASSIGN_SOFT:
      for (int binNum=0; binNum<orientQuant/2+1; ++binNum)
      {
       index2 = (binNum + index + orientQuant - orientQuant/4)%orientQuant;
       binCenter = (index2+.5)*binSize;
       weight = cos(theta - binCenter);
       weight = (weight<0) ? 0 : weight;
       gradMap[index2].at<uchar>(i,j) = (int) round(255*weight);
      }
      break;
     case ASSIGN_SOFT_MAGN:
      for (int binNum=0; binNum<orientQuant/2+1; ++binNum)
      {
       index2 = (binNum + index + orientQuant - orientQuant/4)%orientQuant;
       binCenter = (index2+.5)*binSize;
       weight = cos(theta - binCenter);
       weight = (weight<0) ? 0 : weight;
       gradMap[index2].at<uchar>(i,j) = (int) round(gradMagnitude*weight);
      }
      break;
    }
   }
   ++pDerivy; ++pDerivx;
  }
 }
}

/**********************************************************************************************/
void Utils::computeIntegrals(const vector<Mat>& gradMap,
                             const int orientQuant,
                             vector<Mat>& integralMap)
{
 // Initialize Integral Images
 int rows = gradMap[0].rows;
 int cols = gradMap[0].cols;
 for (int i=0; i<orientQuant+1; ++i)
  integralMap.push_back(Mat::zeros(rows+1, cols+1, CV_8UC1));

 //Generate corresponding integral images;
 for(int i=0; i<orientQuant; ++i)
  integral(gradMap[i],integralMap[i]);

 // copy the values from the first quantization bin
 integralMap[0].copyTo(integralMap[orientQuant]);

 int* ptrSum, *ptr;
 for (int k=1; k<orientQuant; ++k)
 {
  ptr    = (int*) integralMap[k].ptr<int>();
  ptrSum = (int*) integralMap[orientQuant].ptr<int>();
  for (int i=0; i<(rows+1)*(cols+1); ++i)
  {
    *ptrSum += *ptr;
    ++ptrSum;
    ++ptr;
  }
 }
}

/**********************************************************************************************/
float Utils::computeWLResponse(const WeakLearner& WL,
                               const int orientQuant,
                               const vector<Mat>& integralMap)
{
 int width = integralMap[0].cols;
 int idx1 = WL.y_min * width + WL.x_min;
 int idx2 = WL.y_min * width + WL.x_max + 1;
 int idx3 = (WL.y_max + 1) * width + WL.x_min;
 int idx4 = (WL.y_max + 1) * width + WL.x_max + 1;

 int A,B,C,D;
 const int* ptr = integralMap[WL.orient].ptr<int>();
 A=ptr[idx1];
 B=ptr[idx2];
 C=ptr[idx3];
 D=ptr[idx4];
 float current = D+A-B-C;

 ptr = integralMap[orientQuant].ptr<int>();
 A=ptr[idx1];
 B=ptr[idx2];
 C=ptr[idx3];
 D=ptr[idx4];
 float total = D+A-B-C;

 return total ? ( (current / total) - WL.thresh) : 0.f;
 }

/**********************************************************************************************/

void Utils::rectifyPatch(const Mat& image,
                         const KeyPoint& kp,
                         const int& patchSize,
                         Mat& patch)
{
 float s = 1.5f * (float) kp.size / (float) patchSize;

 float cosine = (kp.angle>=0) ? cos(kp.angle*M_PI/180) : 1.f;
 float sine   = (kp.angle>=0) ? sin(kp.angle*M_PI/180) : 0.f;

 float M_[] = {
   s*cosine, -s*sine,   (-s*cosine + s*sine  ) * patchSize/2.0f + kp.pt.x,
   s*sine,   s*cosine,  (-s*sine   - s*cosine) * patchSize/2.0f + kp.pt.y
 };

 warpAffine(image, patch, Mat(2,3,CV_32FC1,M_), Size(patchSize, patchSize),
            CV_WARP_INVERSE_MAP + CV_INTER_CUBIC + CV_WARP_FILL_OUTLIERS );

}

/************************************************************/

double Utils::getMs()
{
   struct timeval t0;
   gettimeofday(&t0, NULL);
   double ret = t0.tv_sec * 1000.0;
   ret += ((double) t0.tv_usec)*0.001;
   return ret;
}

/************************************************************/

bool Utils::isBinary(const string& descriptorType)
{
 return (descriptorType.find("BGM") == 0 ||
         descriptorType.find("BINBOOST") == 0) ?
         true : false;
}

/************************************************************/

int Utils::saveDescriptors(const string& outputFilename,
                    const string& descriptorType,
                    const vector<KeyPoint>& keypoints,
                    const Mat& descriptors)
{
 FILE* output = fopen(outputFilename.c_str(), "w");
 if (!output)
 {
  fprintf(stderr, "[ERROR] Cannot save to %s.\n", outputFilename.c_str());
  return -1;
 }

 fprintf(output, "%d\n%d\n", (int) descriptors.rows, descriptors.cols);
 for (int i=0; i<descriptors.rows; ++i)
 {
   fprintf(output, "%3.2f %3.2f %3.2f %3.2f ",
           keypoints[i].pt.x,
           keypoints[i].pt.y,
           keypoints[i].size,
           keypoints[i].angle);
   if (isBinary(descriptorType))
   {
    const uchar* desc_i = descriptors.ptr<uchar>(i);
    for (int j=0; j<descriptors.cols; ++j)
     fprintf(output, "%d ", desc_i[j]);
   }
   else
   {
    const float* desc_i = descriptors.ptr<float>(i);
    for (int j=0; j<descriptors.cols; ++j)
     fprintf(output, "%3.2f ", desc_i[j]);
   }
   fprintf(output, "\n");
 }
 fclose(output);

 return 0;
}

/************************************************************/

void Utils::saveMatchedImages(const string& img1Filename,
                       const string& img2Filename,
                       const vector<KeyPoint>& keypoints1,
                       const vector<KeyPoint>& keypoints2,
                       const vector<DMatch>& homography_matches,
                       const Mat& H,
                       const string& outputImgFile)
{
 Mat image1 = imread(img1Filename, CV_LOAD_IMAGE_GRAYSCALE);
 Mat image2 = imread(img2Filename, CV_LOAD_IMAGE_GRAYSCALE);

 // get the corners from the object image
 vector<Point2f> obj_corners(4);
 obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
 obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
 vector<Point2f> scene_corners(4);

 perspectiveTransform( obj_corners, scene_corners, H);

 // draw matches
 Mat img_matches;
 drawMatches( image1, keypoints1, image2, keypoints2,
              homography_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
              vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

 // draw lines between the corners (the mapped object in the scene - image )
 line( img_matches, scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar(0, 255, 0), 4 );
 line( img_matches, scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
 line( img_matches, scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
 line( img_matches, scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );

 // save (or show) detected matches
 imwrite(outputImgFile.c_str(), img_matches);
 #if VERBOSE
 fprintf(stdout, "[INFO] Matches saved to '%s'.\n", outputImgFile.c_str());
 #endif
}

/************************************************************/

int Utils::matchFloat(const Mat& descriptors1,
                 const vector<Mat>& descriptors2,
                 vector< vector<DMatch> >& matches)
{
 const float maxDist = std::numeric_limits<float>::max();
 matches.resize(descriptors1.rows);
 #pragma omp parallel for
 for (int i=0; i<descriptors1.rows; ++i)
 {
  float minDist1 = maxDist, minDist2 = maxDist;
  int minId1 = -1, minId2 = -1;
  int minIdImg1 = -1, minIdImg2 = -1;
  const float* descr_i = descriptors1.ptr<float>(i);
  for (unsigned int imgIdx=0; imgIdx<descriptors2.size(); ++imgIdx)
  {
   for (int j=0; j<descriptors2[imgIdx].rows; ++j)
   {
    float currentDist = 0.f;
    const float* descr_j = descriptors2[imgIdx].ptr<float>(j);
    for (int d=0; d<descriptors1.cols; ++d)
    {
     float diff = descr_i[d]-descr_j[d];
     currentDist += diff*diff;
    }
    currentDist=sqrt(currentDist);
    if (currentDist<minDist1)
    {
     minDist2 = minDist1;
     minId2 = minId1;
     minIdImg2 = minIdImg1;
     minDist1 = currentDist;
     minId1 = j;
     minIdImg1 = imgIdx;
    }
    else if (currentDist>=minDist1 && currentDist<minDist2)
    {
     minDist2 = currentDist;
     minId2 = j;
     minIdImg2 = imgIdx;
    }
   }
  }
  #pragma omp critical
  {
   matches[i].push_back(DMatch(i, minId1, minIdImg1, minDist1));
   matches[i].push_back(DMatch(i, minId2, minIdImg2, minDist2));
  }
 }
 return 0;
}

/************************************************************/

int Utils::matchFloat(const Mat& descriptors1,
               const Mat& descriptors2,
               vector< vector<DMatch> >& matches)
{
 vector<Mat> temp;
 temp.push_back(descriptors2);
 return matchFloat(descriptors1, temp, matches);
}


/************************************************************/

int Utils::matchHamming(const Mat& descriptors1,
                 const vector<Mat>& descriptors2,
                 vector< vector<DMatch> >& matches)
{
 const int maxDist = descriptors1.cols<<3;
 matches.resize(descriptors1.rows);
 #pragma omp parallel for
 for (int i=0; i<descriptors1.rows; ++i)
 {
  int minDist1 = maxDist, minDist2 = maxDist;
  int minId1 = -1, minId2 = -1;
  int minIdImg1 = -1, minIdImg2 = -1;
  const long long* descr_i = descriptors1.ptr<long long>(i);
  for (unsigned int imgIdx=0; imgIdx<descriptors2.size(); ++imgIdx)
  {
   for (int j=0; j<descriptors2[imgIdx].rows; ++j)
   {
    int currentDist = 0;
    const long long* descr_j = descriptors2[imgIdx].ptr<long long>(j);
    for (int d=0; d<descriptors1.cols/8; ++d)
    {
     currentDist += __builtin_popcountll(descr_i[d]^descr_j[d]);
    }
    if (currentDist<minDist1)
    {
     minDist2 = minDist1;
     minId2 = minId1;
     minIdImg2 = minIdImg1;
     minDist1 = currentDist;
     minId1 = j;
     minIdImg1 = imgIdx;
    }
    else if (currentDist>=minDist1 && currentDist<minDist2)
    {
     minDist2 = currentDist;
     minId2 = j;
     minIdImg2 = imgIdx;
    }
   }
  }
  #pragma omp critical
  {
   matches[i].push_back(DMatch(i, minId1, minIdImg1, (float) minDist1));
   matches[i].push_back(DMatch(i, minId2, minIdImg2, (float) minDist2));
  }
 }
 return 0;
}

/************************************************************/

int Utils::matchHamming(const Mat& descriptors1,
                 const Mat& descriptors2,
                 vector< vector<DMatch> >& matches)
{
 vector<Mat> temp;
 temp.push_back(descriptors2);
 return matchHamming(descriptors1, temp, matches);
}

}
