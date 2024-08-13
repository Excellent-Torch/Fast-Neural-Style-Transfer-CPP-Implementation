// FastNeuralStyleTransfer.cpp : Defines the entry point for the application.
#include <iostream>
#include <map>
#include <AtlBase.h>
#include <atlconv.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <onnxruntime_cxx_api.h>
#include "FastNeuralStyleTransfer.h"


using namespace std;

//constexpr const int width_ = 1280;
//constexpr const int height_ = 720;
constexpr const int channel = 3;

const map<string, STYLE_TRANSFER_TYPE> choiceOptions = { {"1", STYLE_TRANSFER_TYPE::CARTOON},
{"2", STYLE_TRANSFER_TYPE::CANDY}
};


FastNeuralStyleTransfer::FastNeuralStyleTransfer() : session{ nullptr }
{
   
    

}

cv::Mat FastNeuralStyleTransfer::process(cv::Mat image, int width_, int height_)
{
    

    Ort::AllocatorWithDefaultOptions allocator;

    std::size_t inputCount = session.GetInputCount();

    std::size_t outputCount = session.GetOutputCount();

    std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
    std::vector<const char*> inputNames;

    inputNamesPtr.reserve(inputCount);
    inputNames.reserve(inputCount);

    for (size_t i = 0; i < inputCount; i++)
    {
        auto inputName = session.GetInputNameAllocated(i, allocator);
        inputNames.push_back(inputName.get());
        inputNamesPtr.push_back(std::move(inputName));
    }

    std::vector<Ort::AllocatedStringPtr> outputNamesPtr;
    std::vector<char*> outputNames;
    for (size_t i = 0; i < outputCount; i++)
    {
        auto outputName = session.GetOutputNameAllocated(i, allocator);
        outputNames.push_back(outputName.get());
        outputNamesPtr.push_back(std::move(outputName));
    }

    std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Size originalSize(image.cols, image.rows);
    cv::resize(image, image, cv::Size(width_, height_));

    std::vector<float> imgData;
    imgData.resize(width_ * height_ * channel);

    for (int c = 0; c < channel; ++c)
    {
        for (int i = 0; i < height_; ++i)
        {
            for (int j = 0; j < width_; ++j)
            {
                imgData[c * height_ * width_ + i * width_ + j] = image.at<cv::Vec3b>(i, j)[c] / 255.0f;
            }
        }
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    std::vector<cv::Mat> outputImages;

    for (auto& tensor : outputTensors)
    {
        if (tensor.IsTensor())
        {
            std::vector<int64_t> dimensions = tensor.GetTensorTypeAndShapeInfo().GetShape();
            float* data = tensor.GetTensorMutableData<float>();

            std::vector<cv::Mat> channels;
            int channel_size = dimensions[2] * dimensions[3];
            for (int i = 0; i < dimensions[1]; ++i)
            {
                channels.push_back(cv::Mat(cv::Size(dimensions[3], dimensions[2]), CV_32F, data + i * channel_size));
            }

            cv::Mat mat;
            cv::merge(channels, mat);


            cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
            cv::resize(mat, mat, originalSize);

            outputImages.push_back(mat);
        }
    }
    assert(outputImages.size() == 1);
    return outputImages[0];
}

int main(int width_, int height_)
{
    std:cout << "CHOOSE A STYLE TRANSFER TYPE:\n";
    std::cout << "[1] Cartoon\n";
    std::cout << "[2] Candy\n";

    std::string input;
    std::getline(std::cin, input);

    while (choiceOptions.find(input) == choiceOptions.end())
    {
        std::cout << "Invalid choice!\n\n";

        std::cout << "CHOOSE A STYLE TRANSFER TYPE:\n";
        std::cout << "[1] Cartoon\n";
        std::cout << "[2] Candy\n";

        std::getline(std::cin, input);
    }

    STYLE_TRANSFER_TYPE type = choiceOptions.find(input)->second;

    std::string modelPath = "../../../../Models/";
    switch (type)
    {
    case CARTOON:
        modelPath += "WithCAPE2D_E4-CW10-SW60-Seed-42(20K)_V11.onnx";
        width_ = 1280;
        height_ = 720;
        break;
    case CANDY:
        modelPath += "candy-9.onnx";
        width_ = 224;
        height_ = 224;
        break;
    default:
        throw std::runtime_error("Style Transfer type not handled");
        break;
    }

    std::wstring ws_path = std::wstring(CA2W(std::string(modelPath).c_str()));

    Ort::Env env{ ORT_LOGGING_LEVEL_FATAL, "style-transfer" };
    Ort::SessionOptions SessionOptions;

    //SessionOptions.SetIntraOpNumThreads(1);
    //OrtCUDAProviderOptions cuda_options{};
    //SessionOptions.AppendExecutionProvider_CUDA(cuda_options);

    FastNeuralStyleTransfer FSNT;


    FSNT.session = Ort::Session(env, ws_path.c_str(), SessionOptions);

    std::size_t inputCount = FSNT.session.GetInputCount();

    std::string ass = std::to_string(inputCount);

    std::cout << "Session Started!! \n";

    std::string imagePath;
    std::cout << "Provide the path of an image: ";
    std::cin >> imagePath;

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    cv::Mat processsedImage = FSNT.process(image, width_, height_);
    cv::imwrite("../../../../Images/processed.jpg", processsedImage);
    std::cout << "Image Saved! ";


    //cv::cvtColor(processsedImage, processsedImage, cv::COLOR_BGR2RGB);
   // cv::resize(processsedImage, processsedImage, cv::Size(1280, 720));
    //cv::imshow("Processed Image", processsedImage);


    cv::waitKey(0);

    

    return 0; 

   
   
}

