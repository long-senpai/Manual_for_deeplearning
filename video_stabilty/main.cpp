#include "opencv2/opencv.hpp"
#include <iostream>
#include<stdio.h>
#include<unistd.h>
#include <chrono>
#include<thread>
#include<fstream>
#include <vector>
#include "main.h"
#include <opencv2/videoio.hpp> 
using namespace std;
using namespace cv;
int a = 1;

void VideoStabilization::scaleToFixBorder(cv::Mat& frame) {
    cv::Mat T = cv::getRotationMatrix2D(cv::Point2f(frame.cols / 2, frame.rows / 2), 0, 1.3);
    cv::warpAffine(frame, frame, T, frame.size());
}

cv::Mat VideoStabilization::process(cv::Mat& new_frame) {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    cv::Mat resized_frame;
    // std::cout<<new_frame.rows<<std::endl;
    if (new_frame.rows > MAX_HEIGHT) {
        cv::resize(new_frame, resized_frame, cv::Size(new_frame.cols / new_frame.rows * MAX_HEIGHT, MAX_HEIGHT));
        // cv::resize(new_frame, resized_frame, cv::Size(int(new_frame.cols / 2) , int(new_frame.rows / 2)));
        // resized_frame = new_frame;
    }
    else {
        resized_frame = new_frame;
    }
    if (prev_gray_frame.empty()) {
        cv::cvtColor(resized_frame, prev_gray_frame, cv::COLOR_BGR2GRAY);
        return new_frame;
    }

    cv::Mat gray_frame;
    cv::cvtColor(resized_frame, gray_frame, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> prev_points;

    cv::goodFeaturesToTrack(prev_gray_frame, prev_points, max_corners, corners_quality_level, 30);
    
    if (!prev_points.empty()) {
        std::vector<cv::Point2f> next_points;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_points, next_points, status, err);
        size_t s = 0;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                prev_points[s] = prev_points[i];
                next_points[s] = next_points[i];
                s++;
            }
        }
        prev_points.resize(s);
        next_points.resize(s);
        if (!prev_points.empty() && !next_points.empty()) {
            latest_transform = estimateAffinePartial2D(prev_points, next_points);
        }
    }

    prev_gray_frame = gray_frame;

    if (latest_transform.empty()) {
        return new_frame;
    }

    Trajectory trans_param(
        latest_transform.at<double>(0, 2),
        latest_transform.at<double>(1, 2),
        atan2(latest_transform.at<double>(1, 0), latest_transform.at<double>(0, 0))
    );
    trajectories.push_back(trajectories.empty() ? trans_param : trajectories.back() + trans_param);

    if (trajectories.size() >= used_frame_num) {
        frames.push_back(resized_frame);
    }

    if (trajectories.size() > used_frame_num) {
        transforms.push_back(trans_param);
    }

    if (trajectories.size() < used_frame_num * 2 + 1) {
        return new_frame;
    }

    Trajectory smoothed_trajectory(0, 0, 0);
    for (const Trajectory& t : trajectories) {
        smoothed_trajectory += t;
    }
    smoothed_trajectory /= trajectories.size();
    smoothed_trajectory -= trajectories[used_frame_num];
    smoothed_trajectory += transforms.front();

    cv::Mat stabilized_frame;
    cv::warpAffine(frames.front(), stabilized_frame, smoothed_trajectory.getTransform(), frames.front().size());
    // cv::resize(stabilized_frame, stabilized_frame,)
    scaleToFixBorder(stabilized_frame);
    cv::Mat kernel3 = (Mat_<double>(3,3) <<-1, -1, -1, -1, 9, -1, -1, -1, -1);
    cv::filter2D(stabilized_frame, stabilized_frame, -1, kernel3);

    if (stabilized_frame.size() != new_frame.size()) {
        cv::resize(stabilized_frame, stabilized_frame, new_frame.size(), cv::INTER_CUBIC);
        // cv::bilateralFilter(stabilized_frame,stabilized_frame,9,75,75);
        std::cout<<"a"<<std::endl;
    }
    frames.pop_front();
    transforms.pop_front();
    trajectories.pop_front();
    std::chrono::steady_clock::time_point now1 = std::chrono::steady_clock::now();
    float time_sec = std::chrono::duration<double>(now - now1).count();
    // std::cout<<time_sec<<std::endl;
    return stabilized_frame;
}

int main(int argc, const char** argv) {
    bool stable_Video = true;
    VideoStabilization* stabilizer = nullptr;
    stabilizer = new VideoStabilization(5, 1.1, 200, 0.1);
    
    cv::VideoCapture video_cap("filesrc location=/home/long/Desktop/manual/tanks.mov ! decodebin ! autovideoconvert ! appsink",cv::CAP_GSTREAMER);
    // cv::VideoCapture video_cap("/home/long/Desktop/manual/tanks.mov",cv::CAP_FFMPEG);
    if (!video_cap.isOpened()) {
		std::cout << "Error opening the video source!\n";
		return -1;
	}
    int width = video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // VideoWriter videow("out.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 20, cv::Size(width,height));
    cv::Mat frame_1;
    while (1)
    {
        cv::Mat first_frame;
	video_cap >> first_frame;
	if (first_frame.empty()) {
		std::cout << "Error reading from the video source!\n";
		return -1;
	}
    cv::Mat cap_frame; 
    if (stable_Video){
        cap_frame = stabilizer->process(first_frame);
    }
    
    Mat canvas = Mat::zeros(first_frame.rows, first_frame.cols*2+10, first_frame.type());

        first_frame.copyTo(canvas(Range::all(), Range(0, cap_frame.cols)));

        cap_frame.copyTo(canvas(Range::all(), Range(cap_frame.cols+10, cap_frame.cols*2+10)));

        if(canvas.cols > 1920)
        {
            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
        }
        imshow("before and after", canvas);
        cv::waitKey(1);
        
    }
    video_cap.release();
    // videow.release();
    destroyAllWindows();

    return 0;
}
