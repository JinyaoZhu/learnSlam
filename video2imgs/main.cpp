#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip> 

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace std;

std::string convertDouble(double value) {
  std::ostringstream o;
  if (!(o << value))
    return "";
  return o.str();
}



int main( int argc, char** argv )
{  
  if(argc < 3)
  {
    cout<< "input: directory_of_the_video   output_dir "<<endl;
    return 1;
  }
   
  cv::VideoCapture video(argv[1]);
  
  string output_dir = argv[2];
  
  if(!video.isOpened())
  {
    cout << "can't open video file!" <<endl;
    return 1;
  }
  
  cv::Mat frame,f;
  double scale = 1.0;
  
  int index = 0;
  double timestamp = 0;
  double fps = video.get(CV_CAP_PROP_FPS) ;
  
  ofstream rgb_txt;
   
  rgb_txt.open("./rgb.txt");
  
  for(;;)
  {
    video >> f;
    if(f.empty())  break;
    else frame = f;
    
    index++;
    timestamp += 1.0/fps;
    
//    cout << "time:"<<timestamp <<"s"<<endl;
    
//    cv::resize(frame, frame, cv::Size(frame.size().width*scale,frame.size().height*scale));
     cv::imshow("video",frame);
     cv::waitKey(1);
    stringstream ss;
    ss << index;
    string file_name = ss.str() + ".png";
    
    if(cv::imwrite(output_dir + file_name,frame) != true)
    {
      cout << "imwrite error !"<<endl;
      return 1;
    }
    else
    {
      cout<<"write "<< output_dir+ file_name << " "<<"size:"<< frame.size().width << "x"<<frame.size().height<<endl;
    }
    
    rgb_txt <<convertDouble(timestamp)<<" " <<"rgb/"<<file_name << endl;
  }
  
  cout << "frame rate:"<<fps<<"fps"<<endl;
  //cout << "Frame size:"<<frame.size().width<<"x"<<frame.size().height<<endl;
  video.release();
  rgb_txt.close();

  return 0;
}
