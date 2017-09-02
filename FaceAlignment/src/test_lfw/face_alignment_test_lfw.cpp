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

//#define SAVE_IMAGE
//#define SHOW_IMAGE

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

seeta::Rect LFW_GT_RECT = {68, 68, 120, 120};
float GT_IOU_THRESH = 0.3;

void pring_usage()
{
	cout << "==============" << endl;
	cout << "Usage: fa_test_lfw <image_list_file> [<image_root_dir>]" << endl;
	cout << "       Press 'q' to exit." << endl;
	cout << "==============" << endl;
}

float calc_overlap_iou(seeta::Rect &r1, seeta::Rect &r2)
{
	float s1 = r1.width * r1.height;
	float s2 = r2.width * r2.height;

	float x = std::max<float>(r1.x, r2.x);
	float y = std::max<float>(r1.y, r2.y);
	float w = std::min<float>(r1.x + r1.width, r2.x + r2.width) - x + 1;
	float h = std::min<float>(r1.y + r1.height, r2.y + r2.height) - y + 1;

	if (w <= 0 || h <= 0)
	{
		return 0.0f;
	}
	else
	{
		float s = w*h;
		float iou = s / (s1 + s2 - s);
		return iou;
	}	
}

char json_str[1024];
char json_tmpl[] = 
"{\n"
"  \"face_count\": 1,\n"
"  \"message\" : \"success\",\n"
"  \"faces\" : [\n"
"    {\n"
"      \"score\": %5.3f,\n"
"      \"pts\" : [%5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f],\n"
"      \"rect\": [%d, %d, %d, %d]\n"
"    }\n"
"   ],\n"
"  \"id\": \"%d\", \n"
"  \"filename\" : \"%s\"\n"
"}";


int main(int argc, char** argv)
{
	string image_root_dir = "";
	string fn_lfw_list = "./lfw_list_mtcnn.txt";

	string fn_rlt = "./lfw_seeta_fd_rlt.json";
	ofstream fs_rlt(fn_rlt);

	string fn_log = "./lfw_seeta_fd_log.txt";
	ofstream fs_log(fn_log);

	int first_flag = 1;
	fs_rlt << '[' << endl;
	
	pring_usage();

	if (argc > 1)
	{
		fn_lfw_list = argv[1];
	}

	if (argc > 2)
	{
		image_root_dir = argv[2];
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
	int id;
  
	string line;

	int cnt_fd = 0;
	int cnt_fa = 0;

	int cnt_missed_1 = 0;
	int cnt_missed_2 = 0;

	double fd_time_ttl = 0.0f;
	double fa_time_ttl = 0.0f;

	while (getline(list_fs, line))
	{
		if (line.length() < 3)
		{
			continue;
		}

		istringstream iss(line);

		if (!(iss >> img_fn >> id))
		{
			continue;
		}

		if (first_flag)
		{
			first_flag = 0;
		}
		else
		{
			fs_rlt << "," << endl;
		}

		//load image
		IplImage *img_grayscale = NULL;
		string full_img_path = image_root_dir + "/" + img_fn;

		cout << "Processing image: " << full_img_path << " with id: " << id << endl;

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

		std::cout << "Face Detections takes " << secs << " seconds " << std::endl;

		int32_t face_num = static_cast<int32_t>(faces.size());

		seeta::Rect bbox = LFW_GT_RECT;
		double score = 100.0f;

		if (face_num > 0)
		{
			cout << face_num << " faces detected" << endl;

			int best_idx = 0;
			float best_iou = calc_overlap_iou(faces[0].bbox, LFW_GT_RECT);
			
			for (int i = 1; i < faces.size(); i++)
			{
				float iou = calc_overlap_iou(faces[i].bbox, LFW_GT_RECT);
				if (iou > best_iou)
				{
					best_idx = i;
					best_iou = iou;
				}
			}

			cout << "Best IOU with GT rect is " << best_iou << endl;

			if (best_iou > GT_IOU_THRESH)
			{
				bbox = faces[best_idx].bbox;
				score = faces[best_idx].score;
			}
			else
			{
				cout << "But none of them has IOU>" << GT_IOU_THRESH << " with GT rect" << endl;
			
				cnt_missed_2 += 1;

				fs_log << img_fn << endl;
				fs_log << "But none of them has IOU>" << GT_IOU_THRESH << " with GT rect" << endl;
			}
		}
		else
		{
			cout << "No faces detected, use default GT rect" << endl;

			cnt_missed_1 += 1;
			fs_log << img_fn << endl;
			fs_log << "No faces detected, use default GT rect" << endl;
		}

		// Detect 5 facial landmarks
		seeta::FacialLandmark points[5];
		seeta::FaceInfo face;
		face.bbox = bbox;

		t0 = cv::getTickCount();
		point_detector.PointDetectLandmarks(image_data, face, points);
		t1 = cv::getTickCount();
		secs = (t1 - t0) / cv::getTickFrequency();

		cnt_fa += 1;
		fa_time_ttl += secs;
		
		std::cout << "Facial Points Detections takes " << secs << " seconds " << std::endl;

#if defined(SAVE_IMAGE) || defined(SHOW_IMAGE)
		// Visualize the results
		cvRectangle(img_color, cvPoint(bbox.x, bbox.y), cvPoint(bbox.x + bbox.width - 1, bbox.y + bbox.height - 1), CV_RGB(255, 0, 0));
		for (int i = 0; i < pts_num; i++)
		{
			cvCircle(img_color, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
		}
#endif

#ifdef SHOW_IMAGE
		char win_name[] = "image";
		cvNamedWindow(win_name);
		cvShowImage(win_name, img_color);
		char key = cvWaitKey(0);
#endif

		sprintf(json_str, json_tmpl,
			score,
			points[0].x, points[0].y,
			points[1].x, points[1].y,
			points[2].x, points[2].y,
			points[3].x, points[3].y,
			points[4].x, points[4].y,
			bbox.x, bbox.y, bbox.width, bbox.height,
			id,
			img_fn.c_str()
			);
		fs_rlt << json_str;
		
		//cvSaveImage("result.jpg", img_color);

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

	fs_rlt << "]\n" << endl;

	cout << "FD processed " << cnt_fd << " images, takes " << fd_time_ttl << " secs, avg time: " << fd_time_ttl / cnt_fd << "sec/image" << endl;
	cout << "FA processed " << cnt_fa << " faces, takes " << fa_time_ttl << " secs, avg time: " << fa_time_ttl / cnt_fa << "sec/image" << endl;

	cout << cnt_missed_1 << "GT faces missed because no faces detected" << endl;
	cout << cnt_missed_2 << "GT faces missed because low IOU with GT rect" << endl;
	cout << cnt_missed_1 + cnt_missed_2 << "GT faces missed in total" << endl;

	fs_log << "FD processed " << cnt_fd << " images, takes " << fd_time_ttl << " secs, avg time: " << fd_time_ttl / cnt_fd << "sec/image" << endl;
	fs_log << "FA processed " << cnt_fa << " faces, takes " << fa_time_ttl << " secs, avg time: " << fa_time_ttl / cnt_fa << "sec/image" << endl;

	fs_log << cnt_missed_1 << " GT faces missed because no faces detected" << endl;
	fs_log << cnt_missed_2 << " GT faces missed because low IOU with GT rect" << endl;
	fs_log << cnt_missed_1 + cnt_missed_2 << " GT faces missed in total" << endl;

	return 0;
}
