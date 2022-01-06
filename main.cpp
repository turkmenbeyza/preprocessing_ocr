#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>  

using namespace cv;
using namespace std;

/*Turkmen Beyza, 010622*/
int main(int argc, char** argv)
{
    string root_outp_path = "C:\\Users\\turkm\\source\\repos\\\preprocessing_ocr\\output_images\\";
    string root_inp_path = "C:\\Users\\turkm\\source\\repos\\\preprocessing_ocr\\input_images\\";
    //there are 5 input images in root_inp_path
    //results will be saved to root_outp_path
    //you should change paths according to subitable ones

    int index = 0;
    while (index < 5) {

        // Read the image file
        Mat image = imread(root_inp_path + "input" + to_string(index) + ".jpg");
        Mat gray_img, thr_img, blur_img, dilated_img;
        
        // Check for failure
        if (image.empty())
        {
            cout << "Image Not Found!!!" << endl;
            cin.get(); //wait for any key press
            return -1;
        }

        //gamma correction is applied to input image
        //to correct the brightness 
        //gamma value is tuned experimentally
        Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        double gamma = 0.5;
        for (int i = 0; i < 256; ++i)
            p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        Mat img_gam = image.clone();
        LUT(image, lookUpTable, img_gam);

        //convert rgb image into grayscale
        //before binarization
        cvtColor(img_gam, gray_img, COLOR_BGR2GRAY);

        //histogram equalization to stretch out the intensity range
        //for improvement of the contrast in the grayscale image
        Mat img_equ;
        equalizeHist(gray_img, img_equ);

        //adding gaussianblur for removing noise
        Mat img_blur;
        GaussianBlur(img_equ, img_blur, Size(33, 33), 0);

        //adaptive threshold applied for binarization
        //it gives better results than otsu thresholding
        //get inverse of the binary image to make chars as foreground (white)
        //Binarization to make text as foreground and any others as background
        adaptiveThreshold(img_blur, thr_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

        //opening to remove noises
        Mat rect_kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        Mat open_img;
        morphologyEx(thr_img, open_img, MORPH_OPEN, rect_kernel, Point(-1, -1), 1);

        //dilation to detect text area
        rect_kernel = getStructuringElement(MORPH_CROSS, Size(11, 11));
        dilate(open_img, dilated_img, rect_kernel, Point(-1, -1), 15);
        

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        //find convex hulls in image
        findContours(dilated_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<vector<Point> >hull(contours.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            convexHull(contours[i], hull[i]);
        }

        
        RotatedRect textRect;
        double max_area = 0;

        for (size_t i = 0; i < hull.size(); ++i) {

            RotatedRect minRect = minAreaRect(Mat(hull[i]));
            double minRect_area = minRect.size.width * minRect.size.height;
            //determine the largest hull as the text area
            if (minRect_area > max_area) {
                max_area = minRect_area;
                textRect = minRect;
            }
        }

        //find angle of rotated rectangle contains the text
        double angle = textRect.angle;
        Size rect_size = textRect.size;
        if (textRect.size.width < textRect.size.height) {
            angle = 90 + angle;

            rect_size = Size(textRect.size.height, textRect.size.width);
        }


        // rotated rectangle
        Point2f rect_points[4];
        textRect.points(rect_points);

        //Convert them so we can use them in a fillConvexPoly
        Point vertices[4];
        for (int i = 0; i < 4; ++i) {
            vertices[i] = rect_points[i];
        }

        //keep only text convexhulls
        //bitwise_and enables to throw others
        Mat filled_rect(dilated_img.size(), CV_8UC1, Scalar(0, 0, 0));
        fillConvexPoly(filled_rect, vertices, 4, Scalar(255, 255, 255));
        bitwise_and(filled_rect, dilated_img, dilated_img);

        Mat img2;
        //img2 drawing is familiar rectangle.png sample
        //its not required so commented out
        cvtColor(dilated_img, img2, COLOR_GRAY2BGR);

        Mat img_copy = image.clone();
        //draw rectangle surrounding the text into input image
        for (int j = 0; j < 4; j++)
        {
            //
            //line(img2, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 10);
            line(img_copy, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 10);
        }

        circle(img2, textRect.center, 30, Scalar(0, 255, 0), 10);
        //putText(img2, to_string(angle), Point(textRect.center.x, textRect.center.y + 80),
        //    FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2, LINE_AA);
        //imwrite(root_outp_path + "output" + to_string(index) + "_draw.jpg", img2);
        imwrite(root_outp_path + "output" + to_string(index) + "_drawImg.jpg", img_copy);


        //rotate input image according to rotated rectangle's angle
        Mat only_text;
        cvtColor(thr_img, only_text, COLOR_GRAY2BGR);
        Mat rot_mat = getRotationMatrix2D(textRect.center, angle, 1);
        Mat rotated;
        warpAffine(only_text, rotated, rot_mat, only_text.size(), INTER_CUBIC);
        cvtColor(rotated, rotated, COLOR_BGR2GRAY);
        Mat img_crop;
        //Crop rotated image as it contains only text 
        getRectSubPix(rotated, rect_size, textRect.center, img_crop);

        //threshold again for binarize the grayscale output
        double th_val = threshold(img_crop, img_crop, 0, 255, THRESH_OTSU | THRESH_BINARY);

        //dilation to make clear characters
        rect_kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
        dilate(img_crop, img_crop, rect_kernel, Point(-1, -1), 1);
        imwrite(root_outp_path + "output" + to_string(index) + "_crop.jpg", img_crop);
        
        //save the inverse of output, too
        Mat img_inv;
        bitwise_not(img_crop, img_inv);
        imwrite(root_outp_path + "output" + to_string(index) + "_text_inv.jpg", img_inv);

        cout << "input" << to_string(index) << " is processed.\n";
        index++;
    }
    

    return 0;
}