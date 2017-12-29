/*
 * This code was made to forward a image and get the car predictions using the DetectNet
 * it receive as input one image and save result in a kitti's format txt file.
 * Author: Luan
 */


#include "detect_net.hpp"

#include <stdio.h>

#define FNAME_LENGTH 100
using namespace std;

void removeFileExtension(std::string name_src, char *name_dest)
{
	int i, index = 0;
	const char* p_src = name_src.c_str();

	for(i = 0; i < strlen(p_src); i++)
		if(p_src[i] == '.')
			index = i;

	if(index != 0)
	{
		memset(name_dest, '\0', strlen(name_dest));
		memcpy(name_dest, p_src, index);
	}
		
}

int main(int argc, char **argv)
{
	std::string model_file = "deploy.prototxt";
	std::string trained_file = "car.caffemodel";
    printf("test pic name argv[1]= %s\n", argv[1]);
	if (argc < 2)
	{
		printf("Usage %s <file filenames>\n", argv[0]);
		return -1;
	}

	FILE *input_file = fopen(argv[1], "rt");//first parameter is name of picture to detect
	
	if(input_file == NULL)
	{
		exit(printf("Failed to open the file %s\n", argv[1]));
	}

	// use the gpu and first device
	int gpu = 1;
	int device_id = 0;

	// loads the detect net
	// the network layers weights need to be loaded only once.
	DetectNet detectNet(model_file, trained_file, gpu, device_id);
	
	if(!feof(input_file))
	{
		std::string test_file = argv[1];
		printf("while in & Raiden input_file=%s\n",test_file.c_str());
		//char input_file_name[FNAME_LENGTH];

		//Read file name from input_file
		//fscanf(input_file, "%[^\n]\n", input_file_name);
		//printf("Raiden file name print=%s\n", *input_file_name);
		
		cv::Mat img = cv::imread(test_file, -1);
		// forward the image through the network
		std::vector<float> result = detectNet.Predict(img);
		printf("After == img.cols=%d img.rows=%d\n",img.cols, img.rows);
		
		// fix the scale of image
		float correction_x = img.cols / 1250.0;
		float correction_y = img.rows / 380.0;
	
		char output_file_name[FNAME_LENGTH];
		memset(output_file_name, '\0', FNAME_LENGTH);

		removeFileExtension(test_file, output_file_name);
		sprintf(output_file_name, "%s.txt", output_file_name);
		
	    printf("output_file_name=%s\n",output_file_name);
		FILE *output_file = fopen(output_file_name, "w");
	
		for (int i = 0; i < 10; i++)
		{
			// top left
			float xt = result[5*i] * correction_x;
			float yt = result[5*i + 1] * correction_y;
	
			// botton right
			float xb = result[5*i + 2] * correction_x;
			float yb = result[5*i + 3] * correction_y;
	
			float confidence = result[5*i + 4];
			printf("confidence out=%f i=%d\n",confidence, i);
			if (confidence > 0.0)
			{
				fprintf(output_file, "Car 0.00 0 0.00 %.2f %.2f %.2f %.2f 0.00 0.00 0.00 0.00 0.00 0.00 0 %.2f\n", xt, yt, xb, yb, confidence);
				//cv::rectangle(img, cv::Point(xt,yt), cv::Point(xb,yb), cv::Scalar(0,255,0), 2);
			}
		}
	
		/*
		if (img.empty())
		{
			printf("image empty\n");
		}
		else
		{
			cv::imshow("Image", img);
			cv::waitKey(0);
		}
                */
	}

	return 0;
}
