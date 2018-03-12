#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/tracking/tracking.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/aruco.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <fstream>
#include <streambuf>
#include <cmath>
#include <memory>
#include <map>
#include <istream>
#include <utility>
#include <functional>
#include <cstdlib>

using namespace std;
using namespace cv;

Scalar k_pen_color(255,0,255);
Scalar lk_pen_color(0,255,255);
int radius = 4;

cv::Point get_centroid(std::vector<cv::Point> contour)
{
    cv::Point centroid;
    cv::Moments rect_moments = cv::moments(contour);
    centroid.x = (int) ( rect_moments.m10 / rect_moments.m00);
    centroid.y = (int) ( rect_moments.m01 / rect_moments.m00);
    return centroid;
}

int main()
{
    VideoCapture video_stream(0);
    if(!video_stream.isOpened())
        return -1;

    for( ; ; )
    {
        Mat frame;
        video_stream >> frame;
        if(frame.data == nullptr)
            break;

        cv::imshow("Simple finder - In frame" , frame);

        Mat out_frame = frame.clone();
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

        dlib::shape_predictor sp;
        dlib::deserialize("/home/jps/Trained_models/shape_predictor_68_face_landmarks.dat") >> sp;

        dlib::array2d<dlib::rgb_pixel> img;
        dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(frame));

        std::vector<dlib::rectangle> dets = detector(img);

        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            dlib::full_object_detection shape = sp(img, dets[j]);

            std::vector<cv::Point> cv_left_eye;
            std::vector<cv::Point> cv_right_eye;

            for( int l = 36; l < 42; l++ ){
                cv_right_eye.push_back(cv::Point(shape.part(l).x(), shape.part(l).y()));
            }

            for( int l = 42; l < 48; l++ ){
                cv_left_eye.push_back(cv::Point(shape.part(l).x(), shape.part(l).y()));
            }

            cv::Mat drawing_right_eye = cv::Mat::zeros( frame.size(), CV_8UC3 );
            cv::Mat drawing_left_eye = cv::Mat::zeros( frame.size(), CV_8UC3 );

            cv::drawContours( drawing_right_eye, vector<vector<cv::Point> >(1,cv_right_eye), -1, cv::Scalar(255, 255, 255), -1);
            cv::drawContours( drawing_left_eye, vector<vector<cv::Point> >(1,cv_left_eye), -1, cv::Scalar(255, 255, 255), -1);

            cv::bitwise_and(frame, drawing_right_eye, drawing_right_eye);
            cv::bitwise_and(frame, drawing_left_eye, drawing_left_eye);

            cv::drawContours( drawing_right_eye, vector<vector<cv::Point> >(1,cv_right_eye), -1, cv::Scalar(255, 255, 255), 1);
            cv::drawContours( drawing_left_eye, vector<vector<cv::Point> >(1,cv_left_eye), -1, cv::Scalar(255, 255, 255), 1);

            cv::cvtColor(drawing_right_eye, drawing_right_eye,CV_BGR2GRAY);
            cv::cvtColor(drawing_left_eye, drawing_left_eye,CV_BGR2GRAY);

            cv::equalizeHist( drawing_right_eye, drawing_right_eye );
            cv::equalizeHist( drawing_left_eye, drawing_left_eye );

            cv::threshold(drawing_right_eye, drawing_right_eye, 0, 255, CV_THRESH_BINARY |CV_THRESH_OTSU);
            cv::threshold(drawing_left_eye, drawing_left_eye, 0, 255, CV_THRESH_BINARY |CV_THRESH_OTSU);

            Mat drawing_right_eye_(frame.size(), CV_8UC3, Scalar(255,255, 255));
            Mat drawing_left_eye_(frame.size(), CV_8UC3, Scalar(255,255, 255));

            cv::drawContours( drawing_right_eye_, vector<vector<cv::Point> >(1,cv_right_eye), -1, cv::Scalar(0, 0, 0), -1);
            cv::drawContours( drawing_left_eye_, vector<vector<cv::Point> >(1,cv_left_eye), -1, cv::Scalar(0, 0, 0), -1);

            cv::cvtColor(drawing_right_eye_, drawing_right_eye_,CV_BGR2GRAY);
            cv::cvtColor(drawing_left_eye_, drawing_left_eye_,CV_BGR2GRAY);

            cv::bitwise_or(drawing_right_eye, drawing_right_eye_, drawing_right_eye_);
            cv::bitwise_or(drawing_left_eye, drawing_left_eye_, drawing_left_eye_);

            cv::Mat right_pupil =  cv::Scalar::all(255) - drawing_right_eye_;
            cv::Mat left_pupil =  cv::Scalar::all(255) - drawing_left_eye_;

            vector<vector<cv::Point>> right_contours, left_contours;

            cv::findContours(right_pupil, right_contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            cv::findContours(left_pupil, left_contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            cv::Point right_pupil_point;
            double area_ = 0;
            for(auto cont : right_contours){
                double area = cv::contourArea(cont);
                if(area>area_){
                    area_ = area;
                   right_pupil_point=get_centroid(cont);
                }
            }

            area_ = 0;
            cv::Point left_pupil_point;
            for(auto cont : left_contours){
                double area = cv::contourArea(cont);
                if(area>area_){
                    area_ = area;
                    left_pupil_point = get_centroid(cont);
                }
            }

            cv::circle( out_frame,
                        right_pupil_point,
                        radius,
                        k_pen_color,
                        -1);

            cv::circle( out_frame,
                        left_pupil_point,
                        radius,
                        k_pen_color,
                        -1);

        }
        cv::imshow("Simple finder - Out frame" , out_frame);

        if(waitKey(1) == 27)
            break;
    }
}
