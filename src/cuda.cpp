#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

int main(int argc, char** argv) {
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);

    cv::Mat h_img = cv::imread(argv[1]);

    // Output image must be of this format (cubemap):
    //	+----+----+----+
    //	| Y+ | X+ | Y- |
    //	+----+----+----+
    //	| X- | Z- | Z+ |
    //	+----+----+----+

    float sqr = h_img.cols / 4.0;
    int out_cols = sqr * 3;
    int out_rows = sqr * 2;

    cv::Mat h_output(out_rows, out_cols, CV_8UC3);

    cv::cuda::GpuMat d_img, d_output;

    cv::imshow("Original Image", h_img);

    cout << "----- START ----- " << endl;

    auto begin = chrono::high_resolution_clock::now();

    d_img.upload(h_img);
    d_output.upload(h_output);

    // ----- START CUDA -----
    startCUDA(d_img, d_output);

    d_output.download(h_output);

    auto end = std::chrono::high_resolution_clock::now();

    cout << "----- END ----- " << endl;

    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", h_output);
    cv::imwrite("data/output.jpg", h_output);
    cout << "Time: " << diff.count() << endl;
    cv::waitKey();
    return 0;
}
