#include <memory>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>

enum STYLE_TRANSFER_TYPE
{
    CARTOON,
    CANDY
};

class FastNeuralStyleTransfer
{
public:

    FastNeuralStyleTransfer();
    cv::Mat process(cv::Mat image, int width_, int height_);

    Ort::Session session;

private:






};



