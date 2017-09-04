#include "ros/ros.h"
#include "opencv2/core/core.hpp"
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Point.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv.h>
#include "opencv2/opencv.hpp"
#include <list>
#include "std_msgs/Int64.h"

#include "stdlib.h"
#include "time.h"
#include <cmath>
	
static const std::string SUB_TOPIC = "/cv_camera/image_raw";
static const std::string PUB_TOPIC = "/processed_image";

using namespace cv;
using namespace std;
	
namespace cv{
   using std::vector;
}
/** Global variables */
String black_sign_name = "/home/kylesm/catkin_ws/src/450_TD/BlackSign.xml";
CascadeClassifier black_sign_cascade;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub;
  ros::Publisher point_pub;

public:
  ImageConverter(std::string topic)
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
    image_sub_ = it_.subscribe(SUB_TOPIC, 1,
      &ImageConverter::imageCb, this);

    image_pub = it_.advertise(PUB_TOPIC, 1);
    point_pub = nh_.advertise<geometry_msgs::Point>("/object_location", 1);

  }

  ~ImageConverter() {}

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    // Declare opencv mat pointer
    cv_bridge::CvImagePtr cv_ptr;
    // Try to convert recieved sensor_msgs/Image to opencv
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    
    geometry_msgs::Point pointout;

    std::vector<Rect> blacksign;
    Mat frame_gray;

    cvtColor( cv_ptr->image, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

   //-- Detect Sign
    black_sign_cascade.detectMultiScale( frame_gray, blacksign, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100),Size(275, 275));
    int signinhere = 0;

    for( size_t i = 0; i < blacksign.size(); i++ )
    {
        Point center( blacksign[i].x + blacksign[i].width*0.5, blacksign[i].y + blacksign[i].height*0.5 );
        ellipse(cv_ptr->image, center, Size( blacksign[i].width*0.5, blacksign[i].height*0.5), 0, 0, 360, Scalar(255,0,255), 4, 8, 0 );

    }

    //pointout.y = blacksign[0].y;
    //pointout.x = blacksign[0].x;

    image_pub.publish(cv_ptr->toImageMsg());
    //point_pub.publish(pointout);
    ROS_INFO("Image processed and sent");

  }
};

int main(int argc, char** argv)
{
  if( !black_sign_cascade.load( black_sign_name ) ){ printf("--(!)Error loading\n"); return -1; };
  ros::init(argc, argv, "image_converter");
  ImageConverter ic(argv[0]);
  ros::spin();
  return 0;
}

