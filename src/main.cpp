#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace std;

void cudaCubeToEqui(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float sqr);
void cudaEquiToCube(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float sqr);

enum Conversion {
    toCubemap = 0,
    toEquirectangular = 1
};

int main(int argc, char** argv) {
    int conversion = stoi(argv[2]);

    cv::Mat h_img = cv::imread(argv[1]);
    std::string filename = removePath(argv[1]);
    std::string filenameWithoutExt = removeFileExtension(filename);

    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_NORMAL);

    cv::resizeWindow("Original Image", 720, 480);
    cv::resizeWindow("Processed Image", 720, 480);

    float sqr;
    int out_cols, out_rows;
    switch (conversion) {
        case toCubemap:
            sqr = h_img.cols / 4.0;  // We divide by 4 to get the square size. This comes from the unfolded cubemap representation
            out_cols = sqr * 3;
            out_rows = sqr * 2;
            break;
        case toEquirectangular:
            sqr = h_img.cols / 3.0;
            out_cols = sqr * 4;
            out_rows = sqr * 2;
            break;
    }

    cv::Mat h_output(out_rows, out_cols, CV_8UC3);

    cv::cuda::GpuMat d_img, d_output;

    cv::imshow("Original Image", h_img);

    auto begin = chrono::high_resolution_clock::now();

    d_img.upload(h_img);
    d_output.upload(h_output);

    string out_type;
    switch (conversion) {
        case toCubemap:
            out_type = "equi_";
            cout << "Converting " << filename << " from cubemap to equirectangular." << endl;
            cudaEquiToCube(d_img, d_output, sqr);
            break;
        case toEquirectangular:
            out_type = "cube_";
            cout << "Converting " << filename << " from equirectangular to cubemap." << endl;
            cudaCubeToEqui(d_img, d_output, sqr);
            break;
    }

    d_output.download(h_output);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", h_output);

    string current_time = get_current_time_string();
    string path = "data/out/" + out_type + filenameWithoutExt + "_" + current_time + ".jpg";
    cout << "Finished in " << diff.count() << "s. Output saved to: " << path << endl;

    cv::imwrite(path, h_output);
    cv::waitKey();
    return 0;
}
