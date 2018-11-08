
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <optional>
#include <thread>
#include <functional>
#include <stdexcept>
#include <opencv/cv.hpp>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>
#include "Leap.h"

using std::optional;

class leap_painter_listener : public Leap::Listener {
public:
    virtual void onConnect(const Leap::Controller& controller);
    virtual void onFrame(const Leap::Controller& controller);

private:
    void handle_gestures(const Leap::Controller& controller);

    Leap::Frame frame_;
    optional<cv::Point> prev_point_ = std::nullopt;
};

class leap_painter {
public:
    leap_painter(std::string const& imgfile)
        : imgfile_(imgfile), image_{cv::imread(imgfile)}
    {
        if (image_.empty())
            throw std::runtime_error("Could not load image: " + imgfile);

        //controller_.enableGesture(Leap::Gesture::TYPE_SCREEN_TAP);
        controller_.enableGesture(Leap::Gesture::TYPE_KEY_TAP);
        //controller_.enableGesture(Leap::Gesture::TYPE_CIRCLE);
        controller_.addListener(listener_);
    }

    ~leap_painter()
    {
        cv::destroyWindow(window_);
    }

    void run(std::string const& file, int save_key = 's', int exit_key = ESC_KEY)
    {
        //cv::namedWindow(window_, cv::WINDOW_AUTOSIZE);
        //cv::namedWindow(window_, cv::WINDOW_NORMAL | CV_WINDOW_FREERATIO);
        cv::imshow(window_, image_);
        cv::createTrackbar(trackbar_, window_, NULL, 128);
        cv::setTrackbarPos(trackbar_, window_, 12);
        cv::createTrackbar("R", window_, NULL, 255);
        cv::createTrackbar("G", window_, NULL, 255);
        cv::createTrackbar("B", window_, NULL, 255);
        cv::setTrackbarPos("R", window_, 0);
        cv::setTrackbarPos("G", window_, 255);
        cv::setTrackbarPos("B", window_, 0);
        //cv::setMouseCallback(window_, &leap_painter::on_mouse_impl, this);

        for (;;) {
            int key = cv::waitKey(10);

            if (rewrite_pending_ || pointer_pending_)
                show_image();

            if (key == exit_key)
                break;
            if (key == save_key) {
                if (cv::imwrite(file, image_, {cv::IMWRITE_JPEG_QUALITY, 100}))
                    std::cout << "[!] Current image saved: " << file << std::endl;
            }
        }
    }

    /*
     * Those `pending` flags are meant to prevent multithread-related bugs.
     * The HighGUI GTK backend doesn't support multithreaded GUI opeartions.
     * We just show the updated image on the main thread when a pending is available.
     */
    void show_image()
    {
        if (rewrite_pending_)
            cv::imshow(window_, image_);
        else if (pointer_pending_)
            cv::imshow(window_, pointer_buf_);

        rewrite_pending_ = false;
        pointer_pending_ = false;
    }

    /*
     * The leapmotion callback thread doesn't seem to be changed or does but not quite often.
     * Lock for now just in case.
     */
    void draw_line(cv::Point const& p1, cv::Point const& p2)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        bg_color_ = cv::Scalar(
            cv::getTrackbarPos("B", window_),
            cv::getTrackbarPos("G", window_),
            cv::getTrackbarPos("R", window_));

        cv::line(image_, p1, p2, bg_color_, cv::getTrackbarPos(trackbar_, window_), 4);
        rewrite_pending_ = true;
    }

    void draw_pointer(cv::Point const& point)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        pointer_buf_ = image_.clone();
        //cv::circle(pointer_buf_, point, 5, cv::Scalar(0x00, 0x99, 0x00));
        cv::drawMarker(pointer_buf_, point, cv::Scalar(0x00, 0x99, 0x00), cv::MarkerTypes::MARKER_CROSS, 10, 3, cv::LineTypes::LINE_8);
        pointer_pending_ = true;
    }

    void reset_image()
    {
        image_ = cv::imread(imgfile_);
        rewrite_pending_ = true;
    }

    cv::Mat const& image() const
    {
        return image_;
    }

private:
    std::mutex mtx_;
    std::string imgfile_;
    cv::Mat image_;
    cv::Mat pointer_buf_;
    bool rewrite_pending_ = false;
    bool pointer_pending_ = false;
    cv::Scalar bg_color_ = {0xff, 0xff, 0xff};
    cv::Point prev_point;
    leap_painter_listener listener_;
    Leap::Controller controller_;

    static const int ESC_KEY = 27;
    const std::string window_ = "CV Painter", trackbar_ = "Line Thickness";

    static void on_mouse_impl(int event, int x, int y, int flag, void *data)
    {
        leap_painter *painter = reinterpret_cast<leap_painter *>(data);
        painter->on_mouse(event, x, y, flag);
    }

    void on_mouse(int event, int x, int y, int flag)
    {
        auto current_point = cv::Point(x, y);

        switch(event) {
        case cv::EVENT_MOUSEMOVE:
            if (!(flag & cv::EVENT_FLAG_LBUTTON))
                break;
        case cv::EVENT_LBUTTONUP:
            draw_line(prev_point, current_point);
        case cv::EVENT_LBUTTONDOWN:
            prev_point = current_point;
        }
    }
};

std::unique_ptr<leap_painter> painter;

void leap_painter_listener::onConnect(const Leap::Controller& controller)
{
    std::cout << "[*] LeapMotion connected" << std::endl;
}

void leap_painter_listener::onFrame(const Leap::Controller& controller)
{
    Leap::Frame frame = controller.frame();
    Leap::InteractionBox ibox = frame.interactionBox();
    Leap::HandList hands = frame.hands();

    if (hands.count() > 0) {
        Leap::Vector point = hands[0].stabilizedPalmPosition();
        Leap::Vector normalized_point = ibox.normalizePoint(point, false);

        auto&& image = painter->image();
        float appX = normalized_point.x * image.cols;
        float appY = (1 - normalized_point.y) * image.rows;

        //std::cout << "[*] Finger coord: (" << appX << "," << appY << ") pinch=" << hands[0].pinchStrength() << std::endl;
        cv::Point current_point(appX, appY);

        handle_gestures(controller);

        if (cv::Rect(0, 0, image.cols, image.rows).contains(current_point)) {
            if (prev_point_ && (hands[0].pinchStrength() * 100) >= 70) {
                //std::cout << "[*] prev=" << *prev_point_ << " current=" << current_point << std::endl;
                painter->draw_line(*prev_point_, current_point);
            } else {
                painter->draw_pointer(current_point);
            }

            prev_point_ = current_point;
        } else {
            prev_point_ = std::nullopt;
        }
    }
}

void leap_painter_listener::handle_gestures(const Leap::Controller& controller)
{
    Leap::Frame frame = controller.frame();

    for (auto&& gesture : frame.gestures()) {
        if (gesture.state() == Leap::Gesture::STATE_STOP) {
            std::cout << "[!] gesture: " << gesture.toString() << " type=" << gesture.type() << std::endl;
            painter->reset_image();
        }
    }
}

int main(int argc, char **argv)
{
    int exit_code = 0;

    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " SRC DEST" << std::endl;
            return 1;
        }

        painter = std::make_unique<leap_painter>(argv[1]);
        painter->run(argv[2]);
    } catch (std::exception& e) {
        std::cerr << "[!] Exception: " << e.what() << std::endl;
        exit_code = -1;
    }

    return exit_code;
}
