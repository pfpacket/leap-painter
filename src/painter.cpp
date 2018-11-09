
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
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

#define RESET_GESTURE Leap::Gesture::Type::TYPE_KEY_TAP
#define ERASER_GESTURE Leap::Gesture::TYPE_SCREEN_TAP

using std::optional;

class leap_painter_listener : public Leap::Listener {
public:
    virtual void onConnect(const Leap::Controller& controller);
    virtual void onFrame(const Leap::Controller& controller);
    void handle_hand_moves(const Leap::Controller& controller, Leap::Frame& frame);

private:
    void handle_gestures(const Leap::Controller& controller, Leap::Frame&);

    Leap::Frame frame_;
    optional<cv::Point> prev_point_ = std::nullopt;
};

class leap_painter {
public:
    std::atomic_int pending_rgb_[3] = {0, 0, 0};

    leap_painter(std::string const& imgfile)
        : imgfile_(imgfile), image_{cv::imread(imgfile)}
    {
        if (image_.empty())
            throw std::runtime_error("Could not load image: " + imgfile);

        paint_layer_ = cv::Mat(image_.rows, image_.cols, image_.type(), cv::Scalar(0, 0, 0));

        controller_.enableGesture(RESET_GESTURE);
        controller_.enableGesture(ERASER_GESTURE);

        controller_.addListener(listener_);
    }

    void enable_camera(int index)
    {
        cap_.open(index);

        next_video_image(true);
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
            int key = cv::waitKey(15);

            apply_pending_rgb();
            next_video_image();
            if (repaint_pending_)
                show_image();

            if (key == exit_key)
                break;
            else if (key == 'v')
                enable_camera(3);
            else if (key == save_key) {
                if (cv::imwrite(file, image_, {cv::IMWRITE_JPEG_QUALITY, 100}))
                    std::cout << "[!] Current image saved: " << file << std::endl;
            }
        }
    }

    void addTrackbarValue(std::string const& trackbar, int diff)
    {
        cv::setTrackbarPos(trackbar, window_, cv::getTrackbarPos(trackbar, window_) + diff);
    }

    void apply_pending_rgb()
    {
        addTrackbarValue("R", pending_rgb_[0]);
        addTrackbarValue("G", pending_rgb_[1]);
        addTrackbarValue("B", pending_rgb_[2]);

        pending_rgb_[0] = 0;
        pending_rgb_[1] = 0;
        pending_rgb_[2] = 0;
    }

    /*
     * `repaint_pending_` is meant to avoid multithread-related problems.
     * The HighGUI GTK backend doesn't support multithreaded GUI opeartions.
     * We just show the updated image on the main thread when a pending is available.
     */
    void show_image()
    {
        cv::imshow(window_, back_buffer_);

        repaint_pending_ = false;
    }

    void draw_line(cv::Point const& p1, cv::Point const& p2)
    {
        auto bg_color_ = eraser_ ? cv::Scalar(0, 0, 0)
            : cv::Scalar(
                cv::getTrackbarPos("B", window_),
                cv::getTrackbarPos("G", window_),
                cv::getTrackbarPos("R", window_));

        cv::line(paint_layer_, p1, p2, bg_color_, cv::getTrackbarPos(trackbar_, window_), 4);
        composite();
    }

    void set_pointer(cv::Point const& point)
    {
        pointer_pos_ = point;
        composite();
    }

    void reset_paint()
    {
        paint_layer_ = cv::Scalar(0, 0, 0);
        composite();
    }

    cv::Mat const& image() const
    {
        return image_;
    }

    bool toggle_eraser()
    {
        bool ret = eraser_;
        eraser_ = !eraser_;

        set_pointer(pointer_pos_);

        return ret;
    }

private:
    std::mutex mtx_;
    std::string imgfile_;
    cv::Mat image_;
    cv::Mat paint_layer_;
    cv::Mat back_buffer_;
    cv::Point pointer_pos_;
    bool repaint_pending_ = false;
    cv::Point prev_point;
    leap_painter_listener listener_;
    Leap::Controller controller_;
    bool eraser_ = false;

    cv::VideoCapture cap_;

    static const int ESC_KEY = 27;
    const std::string window_ = "LeapMotion Painter", trackbar_ = "Line Thickness";

    void composite()
    {
        std::lock_guard<std::mutex> lock(mtx_);

        back_buffer_ = image_ + paint_layer_;

        if (eraser_)
            cv::drawMarker(back_buffer_, pointer_pos_, cv::Scalar(0x00, 0x99, 0x00), cv::MarkerTypes::MARKER_SQUARE, 10, 3, cv::LineTypes::LINE_8);
        else
            cv::drawMarker(back_buffer_, pointer_pos_, cv::Scalar(0x00, 0x99, 0x00), cv::MarkerTypes::MARKER_TRIANGLE_DOWN, 10, 3, cv::LineTypes::LINE_8);

        repaint_pending_ = true;
    }

    void next_video_image(bool init_paint_layer = false)
    {
        if (!cap_.isOpened())
            return;

        cv::Mat buf;
        cap_ >> buf;
        if (!buf.empty()) {
            image_ = buf;

            if (init_paint_layer)
                paint_layer_ = cv::Mat(image_.rows, image_.cols, image_.type(), cv::Scalar(0, 0, 0));

            composite();
        }
    }

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

void leap_painter_listener::onConnect(const Leap::Controller&)
{
    std::cout << "[*] LeapMotion connected" << std::endl;
}

static optional<uint32_t> fingerRoll(Leap::Frame& frame, Leap::Finger::Type finger_type)
{
    auto fingerlist = frame.hands()[1].fingers().extended().fingerType(finger_type);;
    if (fingerlist.count() < 1)
        return std::nullopt;

    return std::abs(fingerlist[0].bone(Leap::Bone::Type::TYPE_INTERMEDIATE).direction().roll()) * 100;
}

static void update_with_finger_roll(Leap::Frame& frame, Leap::Finger::Type type, int index)
{
    if (auto roll = fingerRoll(frame, type)) {
        if (0 <= roll && roll <= 60) {
            painter->pending_rgb_[index] += 1;
        } else if (250 <= roll && roll <= 314) {
            painter->pending_rgb_[index] += -1;
        }
    }
}

void leap_painter_listener::handle_hand_moves(const Leap::Controller&, Leap::Frame& frame)
{
    if (frame.hands().count() >= 2) {
        update_with_finger_roll(frame, Leap::Finger::TYPE_THUMB, 0);
        update_with_finger_roll(frame, Leap::Finger::TYPE_INDEX, 1);
        update_with_finger_roll(frame, Leap::Finger::TYPE_PINKY, 2);
    }
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

        cv::Point current_point(appX, appY);

        handle_hand_moves(controller, frame);

        if (cv::Rect(0, 0, image.cols, image.rows).contains(current_point)) {
            if (prev_point_ && (hands[0].pinchStrength() * 100) >= 70) {
                //std::cout << "[*] prev=" << *prev_point_ << " current=" << current_point << std::endl;
                painter->draw_line(*prev_point_, current_point);
            } else {
                painter->set_pointer(current_point);
            }

            prev_point_ = current_point;
        } else {
            prev_point_ = std::nullopt;
        }

        handle_gestures(controller, frame);
    }
}

void leap_painter_listener::handle_gestures(const Leap::Controller&, Leap::Frame& frame)
{
    for (auto&& gesture : frame.gestures()) {
        if (gesture.state() == Leap::Gesture::STATE_STOP) {
            //std::cout << "[!] gesture: " << gesture.toString() << " type=" << gesture.type() << std::endl;

            switch (gesture.type()) {
            case RESET_GESTURE:
                prev_point_ = std::nullopt;
                painter->reset_paint();
                break;
            case ERASER_GESTURE:
                painter->toggle_eraser();
                break;
            default:
                break;
            }
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
