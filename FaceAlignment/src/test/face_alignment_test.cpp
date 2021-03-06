/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"

#include "face_detection.h"
#include "face_alignment.h"

#define SHOW_IMAGE

//#ifdef _WIN32
//std::string DATA_DIR = "../data/";
//std::string MODEL_DIR = "../model/";
////std::string DATA_DIR = "../../data/";
////std::string MODEL_DIR = "../../model/";
//#else
//std::string DATA_DIR = "./data/";
//std::string MODEL_DIR = "./model/";
//#endif


#ifdef _WIN32
std::string seeta_root_dir = "../../";
#else
std::string seeta_root_dir = "../../";
#endif

std::string DATA_DIR = seeta_root_dir + "FaceAlignment/data/";
std::string MODEL_DIR = seeta_root_dir + "FaceAlignment/model/";

std::string fd_model = seeta_root_dir + "FaceDetection/model/seeta_fd_frontal_v1.0.bin";
std::string fa_model = MODEL_DIR + "seeta_fa_v1.1.bin";

int main(int argc, char** argv)
{

	
  // Initialize face detection model
  seeta::FaceDetection detector(fd_model.c_str());
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  seeta::FaceAlignment point_detector(fa_model.c_str());

  //load image
  IplImage *img_grayscale = NULL;
  img_grayscale = cvLoadImage((DATA_DIR + "image_0001.png").c_str(), 0);
  if (img_grayscale == NULL)
  {
    return 0;
  }

  IplImage *img_color = cvLoadImage((DATA_DIR + "image_0001.png").c_str(), 1);
  int pts_num = 5;
  int im_width = img_grayscale->width;
  int im_height = img_grayscale->height;
  unsigned char* data = new unsigned char[im_width * im_height];
  unsigned char* data_ptr = data;
  unsigned char* image_data_ptr = (unsigned char*)img_grayscale->imageData;
  int h = 0;
  for (h = 0; h < im_height; h++) {
    memcpy(data_ptr, image_data_ptr, im_width);
    data_ptr += im_width;
    image_data_ptr += img_grayscale->widthStep;
  }

  seeta::ImageData image_data;
  image_data.data = data;
  image_data.width = im_width;
  image_data.height = im_height;
  image_data.num_channels = 1;

  // Detect faces
  long t0 = cv::getTickCount();
  std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
  int32_t face_num = static_cast<int32_t>(faces.size());
  long t1 = cv::getTickCount();
  double secs = (t1 - t0) / cv::getTickFrequency();

  std::cout << "Face Detections takes " << secs << " seconds " << std::endl;
  
  if (face_num == 0)
  {
    delete[]data;
    cvReleaseImage(&img_grayscale);
    cvReleaseImage(&img_color);
    return 0;
  }

  // Detect 5 facial landmarks
  seeta::FacialLandmark points[5];

  t0 = cv::getTickCount();
  point_detector.PointDetectLandmarks(image_data, faces[0], points);
  t1 = cv::getTickCount();
  secs = (t1 - t0) / cv::getTickFrequency();

  std::cout << "Facial Points Detections takes " << secs << " seconds " << std::endl;

  printf("Face Info:\n");
  printf("--> score: %5.2f\n", faces[0].score);
  printf("--> bbox (x,y,w,h): (%d, %d, %d, %d)\n", faces[0].bbox.x, faces[0].bbox.y, faces[0].bbox.width, faces[0].bbox.height);
  printf("--> pose (yaw, roll, pitch): (%5.2f, %5.2f, %5.2f)\n", faces[0].yaw, faces[0].roll, faces[0].pitch);
  printf("--> facial points:\n");
  
  // Visualize the results
  cvRectangle(img_color, cvPoint(faces[0].bbox.x, faces[0].bbox.y), cvPoint(faces[0].bbox.x + faces[0].bbox.width - 1, faces[0].bbox.y + faces[0].bbox.height - 1), CV_RGB(255, 0, 0));
  for (int i = 0; i<pts_num; i++)
  {
	printf("\t\t(%5.2f, %5.2f)\n", points[i].x, points[i].y);
    cvCircle(img_color, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
  }
  cvSaveImage("result.jpg", img_color);

#ifdef SHOW_IMAGE
  char win_name[] = "image";
  cvNamedWindow(win_name);
  cvShowImage(win_name, img_color);
  char key = cvWaitKey(0);
  cvDestroyAllWindows();
#endif

  // Release memory
  cvReleaseImage(&img_color);
  cvReleaseImage(&img_grayscale);
  delete[]data;

  return 0;
}
