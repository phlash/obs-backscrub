// Playing with OBS plugins..
//
//
#include <thread>
#include <mutex>
#include <condition_variable>
#include <obs-module.h>
#include <stdio.h>
#include <stdarg.h>
#include "libdeepseg.h"

// Demo setting name & default value
static const char DEMO_SETTING[] = "DemoSetting";
static const int DEMO_DEFAULT = 1;

// debugging
void op_printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("obs-play: ");
    vprintf(fmt, ap);
    fflush(stdout);
    va_end(ap);
}
OBS_DECLARE_MODULE()

// A source, used as a filter..
struct obs_play_filter_t {
    // internal filter state
    calcinfo_t info;
    cv::Mat input;
    cv::Mat mask;
    std::thread tid;
    std::mutex lock;
    std::condition_variable_any cond;
    bool new_frame;
    bool done;
    // filter settings
    int setting;
};
static void obs_play_mask_thread(obs_play_filter_t *filter) {
    op_printf("mask_thread: starting..\n");
    while (!filter->done) {
        // wait for a fresh video frame
        {
            std::lock_guard<std::mutex> hold(filter->lock);
            while (!filter->new_frame)
                filter->cond.wait(filter->lock);
            filter->new_frame = false;
            filter->info.raw = filter->input.clone();
        }
        // run inference
        calc_mask(filter->info);
        // update mask
        {
            std::lock_guard<std::mutex> hold(filter->lock);
            filter->mask = filter->info.mask.clone();
        }
    }
    op_printf("mask_thread: done\n");
}
static int _obs_play_get_setting(obs_data_t *settings) { return (int)obs_data_get_int(settings, DEMO_SETTING); }
static const char *obs_play_get_name(void *type_data) { return "Phlash playing about"; }
static void *obs_play_create(obs_data_t *settings, obs_source_t *source) {
    op_printf("create\n");
    // here we instantiate a new filter, loading all required resources (eg: model file)
    // and setting initial values for filter settings
    auto *filter = new obs_play_filter_t;
    filter->setting = _obs_play_get_setting(settings);
    filter->info.modelname = "/home/phlash/code/backscrub/models/selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite";
    filter->info.threads = 2;
    filter->info.width = 640;
    filter->info.height = 480;
    filter->info.debug = 1;
    filter->info.onprep = filter->info.oninfer = filter->info.onmask = NULL;
    filter->info.caller_ctx = NULL;
    if (!init_tensorflow(filter->info)) {
        op_printf("oops initialising Tensorflow\n");
        delete filter;
        filter = NULL;
    }
    filter->new_frame = false;
    filter->done = false;
    filter->tid = std::thread(obs_play_mask_thread, filter);
    return filter;
}
static obs_properties_t *obs_play_get_properties(void *state) {
    op_printf("get_properties\n");
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_int_slider(props, DEMO_SETTING, "Demonstration setting", -127, 127, 1);
    return props;
}
static void obs_play_get_defaults(obs_data_t *settings) {
    op_printf("get_defaults\n");
    obs_data_set_default_int(settings, DEMO_SETTING, DEMO_DEFAULT);
}
static void obs_play_update(void *state, obs_data_t *settings) {
    obs_play_filter_t *filter = (obs_play_filter_t *)state;
    int val = _obs_play_get_setting(settings);
    op_printf("update: settings=%d->%d\n", filter->setting, val);
    // here we change any filter settings (eg: model used, feathering edges, bilateral smoothing)
    filter->setting = val;
}
static void obs_play_destroy(void *state) {
    obs_play_filter_t *filter = (obs_play_filter_t *)state;
    // stop mask thread
    filter->done = true;
    filter->new_frame = true;
    filter->cond.notify_one();
    filter->tid.join();
    // free memory
    delete filter;
}
static void obs_play_video_tick(void *state, float secs) { }
static obs_source_frame *obs_play_filter_video(void *state, obs_source_frame *frame) {
    obs_play_filter_t *filter = (obs_play_filter_t *)state;
    // here we do the video frame processing
    // First, map data into an OpenCV Mat object, then convert to BGR24 (default OCV format)
    // for calling libdeepseg
    cv::Mat out;
    switch (frame->format) {
    // TODO: more video formats!
    case VIDEO_FORMAT_YUY2:
    {
        // YUV2 (https://www.fourcc.org/pixel-format/yuv-yuy2/) as OpenCV array is 2x8bit channels
        // in OBS it arrives as a single plane of 16bits/pixel
        cv::Mat obs(frame->height, frame->width, CV_8UC2, frame->data[0], frame->linesize[0]);
        std::lock_guard<std::mutex> hold(filter->lock);
        cv::cvtColor(obs, filter->input, CV_YUV2BGR_YUY2);
        filter->new_frame = true;
        filter->cond.notify_one();
        // while we have the lock, grab current mask (if any)
        out = filter->info.mask.clone();
        break;
    }
    default:
        op_printf("filter_video: unsupported frame format: %s\n", get_video_format_name(frame->format));
        return frame;
    }
    // No mask yet?
    if (out.empty())
        return frame;
    // Re-size back to OBS video if required
    if (frame->width != filter->info.width || frame->height != filter->info.height)
        cv::resize(out, out, cv::Size(frame->width, frame->height));
    // Mask the video image, leave the human, green screen the rest
    // feather the edges of the mask.
    for (int row=0; row<out.rows; row+=1) {
        uint8_t *prow = frame->data[0] + frame->linesize[0]*row;
        for (int col=0; col<out.cols; col+=1) {
            int m = (int)(*(out.ptr(row, col)));
            int h = 255-m;
            prow[2*col] = (uint8_t)((int)(prow[2*col])*h/255+m);
            if (!h)
                prow[2*col+1] = 0;  // U=V=0 => green
        }
    }
    return frame;
}
static struct obs_source_info play_src {
    // required
    .id = "obs-play",
    .type = OBS_SOURCE_TYPE_FILTER,             // this displays us in source filter dialog
    .output_flags = OBS_SOURCE_ASYNC_VIDEO,     // this means we only need filter_video()
    .get_name = obs_play_get_name,              // this name is shown in source filter dialog
    .create = obs_play_create,                  // new instance of this filter
    .destroy = obs_play_destroy,                // delete filter instance
    // optional
    .get_defaults = obs_play_get_defaults,      // gets default values of properties (settings)
    .get_properties = obs_play_get_properties,  // defines user-adjustable properties (settings)
    .update = obs_play_update,                  // change the filter settings
    .video_tick = obs_play_video_tick,          // frame duration supplied (not absolute time)
    .filter_video = obs_play_filter_video       // process one input to one output frame
};
bool obs_module_load(void) {
    op_printf("load\n");
    // here we take the opportunity to ensure dependent components (eg: libdeepseg) are loadable
    obs_register_source(&play_src);
    return true;
}
