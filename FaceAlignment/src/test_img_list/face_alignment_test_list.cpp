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

using namespace std;

#define SHOW_IMAGE
#define SAVE_IMAGE

#ifdef SAVE_IMAGE
string save_dir = "./fd_list_rlt";

#ifdef WIN32
#include <direct.h>
//#define mkdir(path) _mkdir(path)
#else
#include <sys/types.h>
#include <sys/stat.h>
#define _mkdir(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif
#endif

//#ifdef _WIN32
//string MODEL_DIR = "../model/";
//#else
//string MODEL_DIR = "./model/";
//#endif
//
//string fd_model = "../../FaceDetection/model/seeta_fd_frontal_v1.0.bin";
//string fa_model = MODEL_DIR + "seeta_fa_v1.1.bin";

#ifdef _WIN32
string seeta_root_dir = "../../";
#else
string seeta_root_dir = "../../";
#endif

string fd_model = seeta_root_dir + "FaceDetection/model/seeta_fd_frontal_v1.0.bin";
string fa_model = seeta_root_dir + "FaceAlignment/model/seeta_fa_v1.1.bin";

void pring_usage()
{
	cout << "==============" << endl;
	cout << "Usage: fa_test_lfw <image_list_file> [<image_root_dir>]" << endl;
	cout << "       Press 'q' to exit." << endl;
	cout << "==============" << endl;
}

int main(int argc, char** argv)
{
	string image_root_dir = "";
	string fn_lfw_list = "./list_img.txt";

	string fn_log = "./seeta_fd_list_log.txt";
	ofstream fs_log(fn_log);
	pring_usage();

#ifdef SAVE_IMAGE
	mkdir(save_dir.c_str());
#endif

	if (argc > 1)
	{
		fn_lfw_list = argv[1];
	}

	if (argc > 2)
	{
		image_root_dir = argv[2];
	}

	if (image_root_dir.size() > 0)
	{
		image_root_dir = image_root_dir + '/';
	}

	cout << "face detection model: " << fd_model << endl;
	cout << "face alignment model: " << fa_model << endl;

	cout << "=============="<< endl;
	
	cout << "Image list file: " << fn_lfw_list << endl;
	cout << "Image root dir: " << image_root_dir << endl;

	// Initialize face detection model
	seeta::FaceDetection detector(fd_model.c_str());
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector(fa_model.c_str());

	fstream list_fs(fn_lfw_list);
	string img_fn;
  
	string line;

	int cnt_fd = 0;
	int cnt_fa = 0;


	double fd_time_ttl = 0.0f;
	double fa_time_ttl = 0.0f;

	while (getline(list_fs, line))
	{
		if (line.length() < 3)
		{
			continue;
		}

		istringstream iss(line);

		if (!(iss >> img_fn))
		{
			continue;
		}

		//load image
		IplImage *img_grayscale = NULL;
		string full_img_path = image_root_dir + img_fn;

		cout << "\n===> Processing image: " << full_img_path << endl;
		fs_log << "\n===> Processing image: " << full_img_path << endl;

		img_grayscale = cvLoadImage(full_img_path.c_str(), 0);
		if (img_grayscale == NULL)
		{
			return 0;
		}

		IplImage *img_color = cvLoadImage(full_img_path.c_str(), 1);
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
		vector<seeta::FaceInfo> faces = detector.Detect(image_data);
		long t1 = cv::getTickCount();
		double secs = (t1 - t0) / cv::getTickFrequency();

		cnt_fd += 1;
		fd_time_ttl += secs;

		cout << "Face Detections takes " << secs << " seconds " << endl;
		fs_log << "Face Detections takes " << secs << " seconds " << endl;

		int32_t face_num = static_cast<int32_t>(faces.size());

		cout << face_num << " faces detected " << endl;
		fs_log << face_num << " faces detected " << endl;

		if (face_num == 0)
		{
			delete[]data;
			cvReleaseImage(&img_grayscale);
			cvReleaseImage(&img_color);
			continue;
		}

		for (int j=0; j<face_num; j++)
		{
			printf("Face Info:\n");
			printf("--> score: %5.2f\n", faces[j].score);
			printf("--> bbox (x,y,w,h): (%d, %d, %d, %d)\n", faces[j].bbox.x, faces[j].bbox.y, faces[j].bbox.width, faces[j].bbox.height);
			printf("--> pose (yaw, roll, pitch): (%5.2f, %5.2f, %5.2f)\n", faces[j].yaw, faces[j].roll, faces[j].pitch);
			printf("--> facial points:\n");

			// Detect 5 facial landmarks
			seeta::FacialLandmark points[5];
			t0 = cv::getTickCount();
			point_detector.PointDetectLandmarks(image_data, faces[j], points);
			t1 = cv::getTickCount();
			secs = (t1 - t0) / cv::getTickFrequency();

			cout << "Facial Points Detections takes " << secs << " seconds " << endl;
			fs_log << "Facial Points Detections takes " << secs << " seconds " << endl;

			cnt_fa += 1;
			fa_time_ttl += secs;

#if defined(SHOW_IMAGE) || defined(SAVE_IMAGE)
			// Visualize the results
			cvRectangle(img_color, cvPoint(faces[j].bbox.x, faces[j].bbox.y), cvPoint(faces[j].bbox.x + faces[j].bbox.width - 1, faces[j].bbox.y + faces[j].bbox.height - 1), CV_RGB(255, 0, 0));
			for (int i = 0; i < pts_num; i++)
			{
				cvCircle(img_color, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
			}
		}
#endif

#ifdef SHOW_IMAGE
		char win_name[] = "image";
		cvNamedWindow(win_name, CV_WINDOW_AUTOSIZE);
		cvShowImage(win_name, img_color);
		char key = cvWaitKey(0);
#endif

#ifdef SAVE_IMAGE
		int p1 = img_fn.rfind('/');
		int p2 = img_fn.rfind('\\');

		p1 = (p1 > p2) ? p1 : p2;
		
		string save_name = save_dir + '/' + img_fn.substr(p1);

		cout << "Save rlt image into " << save_name << endl;
		fs_log << "Save rlt image into " << save_name << endl;

		cvSaveImage(save_name.c_str(), img_color);
#endif

		// Release memory
		cvReleaseImage(&img_color);
		cvReleaseImage(&img_grayscale);
		delete[]data;

#ifdef SHOW_IMAGE

		if (key == 'q' || key == 'Q')
		{
			break;
		}
#endif
	}

#ifdef SHOW_IMAGE
	cvDestroyAllWindows();
#endif

	cout << "FD processed " << cnt_fd << " images, takes " << fd_time_ttl << " secs, avg time: " << fd_time_ttl / cnt_fd << "sec/image" << endl;
	cout << "FA processed " << cnt_fa << " faces, takes " << fa_time_ttl << " secs, avg time: " << fa_time_ttl / cnt_fa << "sec/image" << endl;

	fs_log << "FD processed " << cnt_fd << " images, takes " << fd_time_ttl << " secs, avg time: " << fd_time_ttl / cnt_fd << "sec/image" << endl;
	fs_log << "FA processed " << cnt_fa << " faces, takes " << fa_time_ttl << " secs, avg time: " << fa_time_ttl / cnt_fa << "sec/image" << endl;

	return 0;
}
