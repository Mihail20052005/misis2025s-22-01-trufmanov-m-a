#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace cv;

struct Polygon {
    std::vector<Point2f> points;
};

struct ImageData {
    std::string filename;
    std::vector<Polygon> polygons;
};

float calculate_rotation_angle(const std::vector<Point2f>& points) {
    if (points.size() < 4) return 0.0f;

    RotatedRect rect = minAreaRect(points);
    float angle = rect.angle;

    if (angle < -45.0) {
        angle += 90.0;
    } else if (angle > 45.0) {
        angle -= 90.0;
    }

    return angle;
}

void parse_json(const json& j, std::vector<ImageData>& images) {
    try {
        const auto& img_metadata = j["_via_img_metadata"];

        for (const auto& item : img_metadata.items()) {
            const auto& img_data = item.value();
            ImageData image;
            image.filename = img_data["filename"].get<std::string>();

            for (const auto& region : img_data["regions"]) {
                const auto& shape_attrs = region["shape_attributes"];
                std::string shape_name = shape_attrs["name"].get<std::string>();

                if (shape_name == "polygon") {
                    Polygon poly;
                    auto x_coords = shape_attrs["all_points_x"].get<std::vector<int>>();
                    auto y_coords = shape_attrs["all_points_y"].get<std::vector<int>>();

                    for (size_t i = 0; i < x_coords.size(); ++i) {
                        poly.points.emplace_back(x_coords[i], y_coords[i]);
                    }
                    image.polygons.push_back(poly);
                }
            }

            images.push_back(image);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    }
}

int main() {
    // Чтение JSON
    std::ifstream input_file("/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.cw/testing/points.json");
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }

    json j;
    try {
        input_file >> j;
    } catch (const std::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return 1;
    }

    std::vector<ImageData> images;
    parse_json(j, images);

    for (const auto& image : images) {
        std::cout << "Image: " << image.filename << std::endl;

        for (size_t i = 0; i < image.polygons.size(); ++i) {
            const auto& polygon = image.polygons[i];
            float angle = calculate_rotation_angle(polygon.points);

            std::cout << "  Polygon " << i+1 << " rotation angle: "
                      << angle << " degrees" << std::endl;


        }
    }

    return 0;
}