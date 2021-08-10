#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

struct Trajectory {
    double x;
    double y;
    double a; // angle

    Trajectory(double x, double y, double a) : x(x), y(y), a(a) {};

    cv::Mat getTransform() {
        cv::Mat t(2, 3, CV_64F);
        t.at<double>(0, 0) = cos(a);
        t.at<double>(0, 1) = -sin(a);
        t.at<double>(1, 0) = sin(a);
        t.at<double>(1, 1) = cos(a);
        t.at<double>(0, 2) = x;
        t.at<double>(1, 2) = y;
        return t;
    }

    Trajectory operator+(const Trajectory& t) const {
        return Trajectory(x + t.x, y + t.y, a + t.a);
    }

    void operator+=(const Trajectory& t) {
        x += t.x;
        y += t.y;
        a += t.a;
    }

    void operator-=(const Trajectory& t) {
        x -= t.x;
        y -= t.y;
        a -= t.a;
    }

    void operator/=(double d) {
        if (d != 0) {
            x /= d;
            y /= d;
            a /= d;
        }
    }
};

class VideoStabilization {
private:
    static const int MAX_HEIGHT = 720;

    int used_frame_num, max_corners;
    float scale_factor, corners_quality_level;

    cv::Mat prev_gray_frame, latest_transform;

    std::deque<cv::Mat> frames;
    std::deque<Trajectory> trajectories, transforms;

    void scaleToFixBorder(cv::Mat& frame);

public:
    VideoStabilization(int used_frame_num, float scale_factor, int max_corners, float corners_quality_level)
        : used_frame_num(used_frame_num), scale_factor(scale_factor), max_corners(max_corners), corners_quality_level(corners_quality_level) {
        latest_transform = cv::Mat(2, 3, CV_64F);
    };

    cv::Mat process(cv::Mat& frame);

    void stop() {
        prev_gray_frame.release();
        latest_transform.release();
        frames.clear();
        trajectories.clear();
        transforms.clear();
    }
};

