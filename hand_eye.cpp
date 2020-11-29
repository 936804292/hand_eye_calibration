#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include "dr_calibration.h"


int main_handEye() 
{
	std::cout << "---\n";
	std::cout << "OpenCV version : " << CV_VERSION << '\n';
	std::cout << "Major version : " << CV_MAJOR_VERSION << '\n';
	std::cout << "Minor version : " << CV_MINOR_VERSION << '\n';
	std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << '\n';

	// Read the list of transforms
	std::vector<cv::Mat> R_gripper2base, t_gripper2base;
	std::vector<cv::Mat> R_target2cam, t_target2cam;

	cv::FileStorage fs("E:/Project/CV/Project1/data.yaml", cv::FileStorage::FORMAT_YAML);
	if (fs.isOpened()) 
	{
		cv::FileNode fn = fs.root();
		cv::Rect rotation_rect(0, 0, 3, 3);
		cv::Rect translation_rect(3, 0, 1, 3);
		for (cv::FileNodeIterator it = fn.begin(); it != fn.end(); ++it)
		{
			std::string node_name = (*it).name();
			cv::Mat T;
			fs[node_name] >> T;
			cv::Mat rotation = T(rotation_rect);
			cv::Mat translation = T(translation_rect);
			if (node_name.find("gripper2base") != std::string::npos) {
				R_gripper2base.push_back(rotation);
				t_gripper2base.push_back(translation);
			}
			else if (node_name.find("target2cam") != std::string::npos) {
				R_target2cam.push_back(rotation);
				t_target2cam.push_back(translation);
			}
		}
		std::cout << "---\n";
		std::cout << "Num of gripper2base transforms: " << R_gripper2base.size() << '\n';
		std::cout << "Num of target2cam transforms: " << R_gripper2base.size() << '\n';
		std::cout << "---\n";
		// Calibrate

		//R_gripper2base��t_gripper2base�ǻ�е��ץ������ڻ����˻�����ϵ����ת������ƽ����������Ҫͨ���������˶���������ʾ������ȡ��ز���ת������õ���
		//	R_target2cam �� t_target2cam �Ǳ궨�������˫Ŀ�������ξ����ڽ�������궨ʱ������ȡ�õ���calibrateCamera()�õ���, Ҳ����ͨ��solvePnP()������ȡ����������� �� 
		//	R_cam2gripper �� t_cam2gripper���������۾���ֽ�õ�����ת������ƽ�ƾ���
		//	OpenCVʵ����5�ַ�����ȡ���۾��� �� Tsai�������ٶ����
		cv::Mat R_cam2gripper, t_cam2gripper;
		cv::calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper, t_cam2gripper,cv::CALIB_HAND_EYE_TSAI);
		//  Output the results
		cv::Mat T_cam2gripper = cv::Mat::zeros(4, 4, CV_64F);
		R_cam2gripper.copyTo(T_cam2gripper(rotation_rect));
		t_cam2gripper.copyTo(T_cam2gripper(translation_rect));
		std::cout << "Estimated cam2gripper: \n" << T_cam2gripper << '\n';
	}
	else 
	{
		std::cout << "open file error\n";
	}
	////////////////////////////
	//Subminor version : 0
	//	-- -
	//	Num of gripper2base transforms : 10
	//	Num of target2cam transforms : 10
	//	-- -
	//	Estimated cam2gripper :
	//[0.8357872530342421, 0.4275441928943814, 0.3444787813316921, 30.08659781703732;
	//-0.4108751612084567, 0.90320366951094, -0.1241158059390393, 88.40409398451872;
	//-0.3641994914430832, -0.03780336630863099, 0.9305534030502145, 297.8756595212808;
	//0, 0, 0, 0]

	return 0;
}

int main() 
{
	cv::Size size(7, 9);
	std::string imgPath = "E:/Project/CV/Project1/data/calibration_image";
	std::string calibDataPath = "E:/Project/CV/Project1/data/calibration_image/calib_data.yaml";
	DR_Calibration::Pattern Pat = DR_Calibration::Pattern::CHESSBOARD;
	DR_Calibration calib(imgPath, calibDataPath, size, 0.02, Pat);
	
	calib.doCalibration();



	
}