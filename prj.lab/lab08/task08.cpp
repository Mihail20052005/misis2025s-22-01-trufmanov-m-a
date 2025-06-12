#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

cv::Mat grayWorld(const cv::Mat& image) {
    cv::Mat result;
    image.convertTo(result, CV_32F);
    std::vector<cv::Mat> channels;
    cv::split(result, channels);

    cv::Scalar mean = cv::mean(result);
    double avg = (mean[0] + mean[1] + mean[2]) / 3.0;

    for (int i = 0; i < 3; ++i) {
        double channelMean = cv::mean(channels[i])[0];
        if (channelMean > 0) {
            double scale = avg / channelMean;
            channels[i] = channels[i] * scale;
        }
    }

    cv::merge(channels, result);
    result.convertTo(result, CV_8U);
    return result;
}

cv::Mat colorCorrection(const cv::Mat& image) {
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    cv::Mat lChannel;
    labChannels[0].convertTo(lChannel, CV_8U);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2);
    clahe->setTilesGridSize(cv::Size(8, 8));
    clahe->apply(lChannel, lChannel);

    lChannel.convertTo(labChannels[0], labChannels[0].type());
    cv::merge(labChannels, labImage);
    cv::Mat result;
    cv::cvtColor(labImage, result, cv::COLOR_Lab2BGR);
    return result;
}

cv::Mat visualizeColorDistribution(const cv::Mat& image) {
    std::vector<cv::Mat> bgrChannels;
    cv::split(image, bgrChannels);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::Mat bHist, gHist, rHist;
    cv::calcHist(&bgrChannels[0], 1, 0, cv::Mat(), bHist, 1, &histSize, &histRange);
    cv::calcHist(&bgrChannels[1], 1, 0, cv::Mat(), gHist, 1, &histSize, &histRange);
    cv::calcHist(&bgrChannels[2], 1, 0, cv::Mat(), rHist, 1, &histSize, &histRange);

    int histWidth = 512, histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(bHist, bHist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(gHist, gHist, 0, histImage.rows, cv::NORM_MINMAX);
    cv::normalize(rHist, rHist, 0, histImage.rows, cv::NORM_MINMAX);

    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(bHist.at<float>(i - 1))),
            cv::Point(binWidth * (i), histHeight - cvRound(bHist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2);
        cv::line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(gHist.at<float>(i - 1))),
            cv::Point(binWidth * (i), histHeight - cvRound(gHist.at<float>(i))),
            cv::Scalar(0, 255, 0), 2);
        cv::line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(rHist.at<float>(i - 1))),
            cv::Point(binWidth * (i), histHeight - cvRound(rHist.at<float>(i))),
            cv::Scalar(0, 0, 255), 2);
    }
    return histImage;
}

double calculateMSE(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return -1;
    }
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    cv::Scalar s = cv::sum(diff);
    return s[0] / (img1.total() * img1.channels());
}

double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    double mse = calculateMSE(img1, img2);
    if (mse <= 1e-10) return 100;
    return 10.0 * log10((255.0 * 255.0) / mse);
}

double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    const double C1 = 6.5025, C2 = 58.5225;
    cv::Mat I1, I2;
    img1.convertTo(I1, CV_32F);
    img2.convertTo(I2, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(I1.mul(I1), sigma1_sq, cv::Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;
    cv::GaussianBlur(I2.mul(I2), sigma2_sq, cv::Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;
    cv::GaussianBlur(I1.mul(I2), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_sq + mu2_sq + C1;
    t2 = sigma1_sq + sigma2_sq + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    cv::Scalar mssim = cv::mean(ssim_map);
    return (mssim[0] + mssim[1] + mssim[2]) / 3;
}

void saveQualityParameters(const std::string& imagePath, const cv::Mat& original) {
    const std::string qualityDir = "C:\\Users\\user\\Desktop\\misis2025s-3-nurgaliev-r-d\\prj.lab\\lab08\\color_correction_quality";

    std::string baseName = imagePath.substr(imagePath.find_last_of("/\\") + 1);
    size_t dotPos = baseName.find_last_of('.');
    if (dotPos != std::string::npos) {
        baseName = baseName.substr(0, dotPos);
    }

    std::ofstream outFile(qualityDir + "\\" + baseName + "_quality_parameters.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error creating quality file for " << baseName << std::endl;
        return;
    }

    cv::Mat grayWorldImg = grayWorld(original);
    cv::Mat correctedImg = colorCorrection(original);

    double mse_gw = calculateMSE(original, grayWorldImg);
    double psnr_gw = calculatePSNR(original, grayWorldImg);
    double ssim_gw = calculateSSIM(original, grayWorldImg);

    double mse_clahe = calculateMSE(original, correctedImg);
    double psnr_clahe = calculatePSNR(original, correctedImg);
    double ssim_clahe = calculateSSIM(original, correctedImg);

    outFile << "Quality parameters for: " << baseName << "\n\n";
    outFile << "=== Gray World Correction ===\n";
    outFile << "MSE:  " << mse_gw << "\n";
    outFile << "PSNR: " << psnr_gw << " dB\n";
    outFile << "SSIM: " << ssim_gw << "\n\n";

    outFile << "=== CLAHE Correction ===\n";
    outFile << "MSE:  " << mse_clahe << "\n";
    outFile << "PSNR: " << psnr_clahe << " dB\n";
    outFile << "SSIM: " << ssim_clahe << "\n";

    outFile.close();
}

void processImage(const std::string& path) {
    const std::string resultDir = "/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.lab/result_images";
    const std::string histDir = "/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.lab/histograms";

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Could not open image: " << path << std::endl;
        return;
    }

    cv::Mat grayWorldImg = grayWorld(img);
    cv::Mat correctedImg = colorCorrection(img);

    cv::Mat histOrig = visualizeColorDistribution(img);
    cv::Mat histGW = visualizeColorDistribution(grayWorldImg);
    cv::Mat histCorr = visualizeColorDistribution(correctedImg);

    std::string baseName = path.substr(path.find_last_of("/\\") + 1);
    size_t dotPos = baseName.find_last_of('.');
    if (dotPos != std::string::npos) {
        baseName = baseName.substr(0, dotPos);
    }

    cv::imwrite(resultDir + "\\" + baseName + "_grayworld.jpg", grayWorldImg);
    cv::imwrite(resultDir + "\\" + baseName + "_clahe.jpg", correctedImg);
    cv::imwrite(histDir + "\\" + baseName + "_hist_orig.jpg", histOrig);
    cv::imwrite(histDir + "\\" + baseName + "_hist_gw.jpg", histGW);
    cv::imwrite(histDir + "\\" + baseName + "_hist_clahe.jpg", histCorr);

    saveQualityParameters(path, img);
}

void processImagePair(const std::string& path1, const std::string& path2) {
    cv::Mat img1 = cv::imread(path1);
    cv::Mat img2 = cv::imread(path2);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Could not open images: " << path1 << " or " << path2 << std::endl;
        return;
    }

    cv::resize(img2, img2, img1.size());

    double origMSE = calculateMSE(img1, img2);
    double origPSNR = calculatePSNR(img1, img2);
    double origSSIM = calculateSSIM(img1, img2);

    cv::Mat gw1 = grayWorld(img1);
    cv::Mat gw2 = grayWorld(img2);
    double gwMSE = calculateMSE(gw1, gw2);
    double gwPSNR = calculatePSNR(gw1, gw2);
    double gwSSIM = calculateSSIM(gw1, gw2);

    cv::Mat corr1 = colorCorrection(img1);
    cv::Mat corr2 = colorCorrection(img2);
    double corrMSE = calculateMSE(corr1, corr2);
    double corrPSNR = calculatePSNR(corr1, corr2);
    double corrSSIM = calculateSSIM(corr1, corr2);

    std::cout << "\n=== Comparison Results ===" << std::endl;
    std::cout << "Original images:" << std::endl;
    std::cout << "  MSE: " << origMSE << " | PSNR: " << origPSNR << " dB | SSIM: " << origSSIM << std::endl;
    std::cout << "After Gray World:" << std::endl;
    std::cout << "  MSE: " << gwMSE << " | PSNR: " << gwPSNR << " dB | SSIM: " << gwSSIM << std::endl;
    std::cout << "After CLAHE Correction:" << std::endl;
    std::cout << "  MSE: " << corrMSE << " | PSNR: " << corrPSNR << " dB | SSIM: " << corrSSIM << std::endl;

    processImage(path1);
    processImage(path2);
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc < 2) {
        std::cout << "Usage:\n"
            << "  Single image: " << argv[0] << " image_path\n"
            << "  Compare two:  " << argv[0] << " image1_path image2_path" << std::endl;
        return 1;
    }

    if (argc == 2) {
        processImage(argv[1]);
    }
    else {
        processImagePair(argv[1], argv[2]);
    }

    return 0;
}