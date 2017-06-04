#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>


using namespace std;


int main( int argc, char** argv )
{
  string output_dir;
  
  if(argc < 2)
  {
    cout<< "input: resize list_of_pictures.txt  dir_of_output  scale"<<endl;
    return 1;
  }
  
  ifstream fin (argv[1]);
  output_dir = argv[2];
  
  if ( !fin )
  {
      cout<<"please input the list_of_pictures.txt"<<endl;
      return 1;
  }
  
  vector<string> img_files;
  
  /* read file list */
  while ( !fin.eof() )
  {
      string img;
      fin>>img;
      if ( fin.good() == false )
	  break;
      img_files.push_back (img);
  }
  

  for(auto s : img_files)
  {
    cv::Mat img_src = cv::imread(s);
    cv::Mat img_dist;
    
    double scale = atof(argv[3]);
    cv::Size dsize = cv::Size(img_src.cols*scale,img_src.rows*scale);
    cv::resize(img_src, img_dist, dsize);
    
    vector<string> s_split;
    boost::split( s_split, s, boost::is_any_of( "/" ), boost::token_compress_on );
   
    if(cv::imwrite(output_dir + s_split.back(),img_dist) != true)
    {
      cout << "imwrite error !"<<endl;
      return 1;
    }
    else
    {
      cout<<"write "<< s_split.back() << " "<<"size:"<< img_src.size().width << "x"<<img_src.size().height<<" ===> "
       <<img_dist.size().width<<"x"<< img_dist.size().height << endl;
    }
  }

  return 0;
}
