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

            for( int l = 0; l < shape.num_parts(); l++ )
            {
                cv::circle( out_frame,
                            cv::Point(shape.part(l).x(), shape.part(l).y()),
                            radius,
                            k_pen_color,
                            -1);

                cv::putText(out_frame, std::to_string(l+1),
                            cv::Point(shape.part(l).x(), shape.part(l).y()),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.5 ,
                            lk_pen_color,
                            2);
            }

        }
        cv::imshow("Simple finder - Out frame" , out_frame);

        if(waitKey(1) == 27)
            break;
    }
}
