#include "ros/ros.h"
#include "opencv2/core/core.hpp"
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <visualization_msgs/Marker.h>
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
String black_sign_name = "/home/kyle/catkin_ws/src/SignD/450_TD/BlackSign.xml";
CascadeClassifier black_sign_cascade;
String red_sign_name = "/home/kyle/catkin_ws/src/SignD/450_TD/RedSignHAAR.xml";
CascadeClassifier red_sign_cascade;
String yellow_sign_name = "/home/kyle/catkin_ws/src/SignD/450_TD/YellowSignHAAR.xml";
CascadeClassifier yellow_sign_cascade;

class ImageConverter{
  private:
	ros::NodeHandle nh_;
  	ros::Subscriber sub_camera_info_;

  	image_transport::ImageTransport it_;
  	image_transport::Subscriber image_sub_;
  	image_transport::Publisher image_pub;
  	ros::Publisher point_pub;

  	bool got_camera_info_;
  	bool param_camera_rectified_;

  	sensor_msgs::CameraInfo camera_info_;
  	cv::Mat camera_matrix_;
  	cv::Mat dist_coeffs_;

  	std::string topic_input_camera_info_;

  public:
   	ImageConverter(std::string topic) : 
		it_(nh_)
		//topic_input_camera_info( "input_camera_info" )
	{ 

    		// Subscribe to input video feed and publish output video feed
    		image_sub_ = it_.subscribe(SUB_TOPIC, 1,&ImageConverter::imageCb, this);

    		image_pub = it_.advertise(PUB_TOPIC, 1);
    		point_pub = nh_.advertise<geometry_msgs::Point>("/object_location", 1);

    		sub_camera_info_ = nh_.subscribe<sensor_msgs::CameraInfo> ("/cv_camera/camera_info",1, &ImageConverter::camera_info_cb, this);
    		//topic_input_camera_info_( "input_camera_info" );

  	}
  ~ImageConverter() {}

   //we can use camera info to correct distortions, get image size etc
   void camera_info_cb( const sensor_msgs::CameraInfo::ConstPtr& msg_in ){
	camera_info_ = *msg_in;

	cv::Mat_<double>( msg_in->D ).reshape( 0,1 ).copyTo( dist_coeffs_);

	cv::Mat_<double> m;	
	if( param_camera_rectified_ ) {
		m.push_back( msg_in->P[0] );
		m.push_back( msg_in->P[1] );
		m.push_back( msg_in->P[2] );
		m.push_back( msg_in->P[3] );
		m.push_back( msg_in->P[4] );
		m.push_back( msg_in->P[5] );
		m.push_back( msg_in->P[6] );
		m.push_back( msg_in->P[7] );
		m.push_back( msg_in->P[8] );
		m.push_back( msg_in->P[9] );
		m.push_back( msg_in->P[10] );
	} else {
		for(int i = 0; i < 9; i++) //raw data into matrix
			m.push_back( msg_in->K[i] );
	}
	
	m.reshape(0, 3 ).copyTo( camera_matrix_ ); //reshape to 3x3 

	got_camera_info_ = true;
   }

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
    std::vector<Rect> redsign;
    std::vector<Rect> yellowsign;
    Mat frame_gray;

    cvtColor( cv_ptr->image, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

   //-- Detect Sign
    black_sign_cascade.detectMultiScale( frame_gray, blacksign, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100),Size(275, 275));
    int blacksigninhere = 0;

    for( size_t i = 0; i < blacksign.size(); i++ )
    {
        Point center( blacksign[i].x + blacksign[i].width*0.5, blacksign[i].y + blacksign[i].height*0.5 );
        ellipse(cv_ptr->image, center, Size( blacksign[i].width*0.5, blacksign[i].height*0.5), 0, 0, 360, Scalar(255,0,255), 4, 8, 0 );
	pointout.y = blacksign[0].y + blacksign[0].height*0.5;
        pointout.x = blacksign[0].x + blacksign[0].width*0.5;
    }
/*
    red_sign_cascade.detectMultiScale( frame_gray, redsign, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100),Size(275, 275));
    int redsigninhere = 0;

    for( size_t i = 0; i < redsign.size(); i++ )
    {
        Point center( redsign[i].x + redsign[i].width*0.5, redsign[i].y + redsign[i].height*0.5 );
        ellipse(cv_ptr->image, center, Size( redsign[i].width*0.5, redsign[i].height*0.5), 0, 0, 360, Scalar(255,0,255), 4, 8, 0 );
	pointout.y = redsign[0].y + redsign[0].height*0.5;
        pointout.x = redsign[0].x + redsign[0].width*0.5;
    }

    yellow_sign_cascade.detectMultiScale( frame_gray, yellowsign, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100),Size(275, 275));
  int yellowsigninhere = 0;

    for( size_t i = 0; i < yellowsign.size(); i++ )
    {
        Point center( yellowsign[i].x + yellowsign[i].width*0.5, yellowsign[i].y + yellowsign[i].height*0.5 );
        ellipse(cv_ptr->image, center, Size( yellowsign[i].width*0.5, yellowsign[i].height*0.5), 0, 0, 360, Scalar(255,0,255), 4, 8, 0 );
	pointout.y = yellowsign[0].y + yellowsign[0].height*0.5;
        pointout.x = yellowsign[0].x + yellowsign[0].width*0.5;
    }
*/


    image_pub.publish(cv_ptr->toImageMsg());
    point_pub.publish(pointout);
    ROS_INFO("Image processed and sent");

  }
};

int main(int argc, char** argv)
{
  if( !black_sign_cascade.load( black_sign_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !red_sign_cascade.load( red_sign_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !yellow_sign_cascade.load( yellow_sign_name ) ){ printf("--(!)Error loading\n"); return -1; };
  ros::init(argc, argv, "image_converter");
  ImageConverter ic(argv[0]);
  ros::spin();
  return 0;
}

