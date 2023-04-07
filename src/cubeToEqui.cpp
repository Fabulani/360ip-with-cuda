#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float sqr);

int main(int argc, char** argv) {
    cv::Mat h_img = cv::imread(argv[1]);
    std::string filename = removePath(argv[1]);
    std::string filenameWithoutExt = removeFileExtension(filename);

    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);

    cv::resizeWindow("Original Image", 720, 480);
    cv::resizeWindow("Processed Image", 720, 480);

    /*
    #   Assuming a cubemap input image in the following format:
    #
    #	+----+----+----+
    #	| Y+ | X+ | Y- |
    #	+----+----+----+
    #	| X- | Z- | Z+ |
    #	+----+----+----+
    */

    float sqr = h_img.cols / 3.0;
    int out_cols = sqr * 4;
    int out_rows = sqr * 2;

    cv::Mat h_output(out_rows, out_cols, CV_8UC3);

    cv::cuda::GpuMat d_img, d_output;

    cv::imshow("Original Image", h_img);

    cout << "[cubeToEqui] Converting " << filename << " from cubemap to equirectangular." << endl;

    auto begin = chrono::high_resolution_clock::now();

    d_img.upload(h_img);
    d_output.upload(h_output);

    // ----- START CUDA -----
    startCUDA(d_img, d_output, sqr);

    d_output.download(h_output);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", h_output);

    string current_time = get_current_time_string();
    string path = "data/out/equi_" + filenameWithoutExt + "_" + current_time + ".jpg";
    cout << "[cubeToEqui] Finished in " << diff.count() << "s. Output saved to: " << path << endl;

    cv::imwrite(path, h_output);
    cv::waitKey();
    return 0;
}
