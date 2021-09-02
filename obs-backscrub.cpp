// Simple OBS Studio plugin to use libbackscrub as a background removal filter
//
//
#include <thread>
#include <mutex>
#include <condition_variable>
#include <obs-module.h>
#include <stdio.h>
#include "lib/libbackscrub.h"

// Setting names & default values
static const char MODEL_SETTING[] = "Segmentation model";
static const char MODEL_DEFAULT[] = "selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite";
static const size_t BS_THREADS = 2;
static const size_t BS_WIDTH = 640;
static const size_t BS_HEIGHT = 480;

// debugging
static void obs_backscrub_dbg(void *ctx, const char *msg) {
    blog(LOG_INFO, "obs-backscrub(%p): %s", ctx, msg);
}
static void obs_printf(void *ctx, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *msg;
    if (vasprintf(&msg, fmt, ap)) {
        obs_backscrub_dbg(ctx, msg);
        free(msg);
    }
    va_end(ap);
}

OBS_DECLARE_MODULE()

// A source, used as a filter..
struct obs_backscrub_filter_t {
    // internal filter state
    void *maskctx;
    char *modelname;
    size_t width;
    size_t height;
    cv::Mat input;
    cv::Mat mask;
    std::thread tid;
    std::mutex lock;
    std::condition_variable_any cond;
    bool new_frame;
    bool done;
    // additional blend settings
};
static void obs_backscrub_mask_thread(obs_backscrub_filter_t *filter) {
    obs_printf(filter, "mask_thread: starting..");
    while (!filter->done) {
        // wait for a fresh video frame
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> hold(filter->lock);
            while (!filter->new_frame)
                filter->cond.wait(filter->lock);
            filter->new_frame = false;
            frame = filter->input.clone();
        }
        // check for empty frame (can happen if we are terminated before video starts)
        if (frame.empty())
            continue;
        // run inference
        cv::Mat mask;
        bs_maskgen_process(filter->maskctx, frame, mask);
        // update mask
        {
            std::lock_guard<std::mutex> hold(filter->lock);
            filter->mask = mask;
        }
    }
    obs_printf(filter, "mask_thread: done");
}
static char *_obs_backscrub_get_model(obs_data_t *settings) {
    const char *settings_path = obs_data_get_string(settings, MODEL_SETTING);
    char *rv = nullptr;
    // relative paths, map through module location
    if (settings_path[0] == '/')
        rv = bstrdup(settings_path);
    else
        rv = obs_module_file(settings_path);
    if (!rv)
        obs_printf(nullptr, "_get_path: NULL file mapping, maybe missing module data folder?");
    return rv;
}
static const char *obs_backscrub_get_name(void *type_data) { return "Background scrubber"; }
static void *obs_backscrub_create(obs_data_t *settings, obs_source_t *source) {
    // here we instantiate a new filter, loading all required resources (eg: model file)
    // and setting initial values for filter settings
    auto *filter = new obs_backscrub_filter_t;
    obs_printf(filter, "create");
    filter->modelname = _obs_backscrub_get_model(settings);
    filter->width = BS_WIDTH;
    filter->height = BS_HEIGHT;
    filter->maskctx = bs_maskgen_new(filter->modelname, BS_THREADS, filter->width, filter->height,
        obs_backscrub_dbg, nullptr, nullptr, nullptr, nullptr);
    if (!filter->maskctx) {
        obs_printf(filter, "oops initialising backscrub");
        // if creation failed we still need to return a state,
        // otherwise the user won't be able to fix the config.
        return filter;
    }
    filter->new_frame = false;
    filter->done = false;
    filter->tid = std::thread(obs_backscrub_mask_thread, filter);
    obs_printf(filter, "create: done");
    return filter;
}
static void obs_backscrub_get_defaults(obs_data_t *settings) {
    obs_printf(nullptr, "get_defaults");
    obs_data_set_default_string(settings, MODEL_SETTING, MODEL_DEFAULT);
}
static obs_properties_t *obs_backscrub_get_properties(void *state) {
    obs_printf(nullptr, "get_properties");
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_path(props, MODEL_SETTING, "Segmentation model file", OBS_PATH_FILE,
        "TFLite models (*.tflite)", MODEL_DEFAULT);
    return props;
}
static void obs_backscrub_update(void *state, obs_data_t *settings) {
    obs_backscrub_filter_t *filter = (obs_backscrub_filter_t *)state;
    char *model = _obs_backscrub_get_model(settings);
    obs_printf(filter, "update: model: %s=>%s", filter->modelname, model);
    // here we change any filter settings (eg: model used, feathering edges, bilateral smoothing)
    if (filter->modelname == model) return; // same pointer === same string
    if (!filter->modelname || !model || strcmp(model, filter->modelname)) {
        // stop mask thread
        filter->done = true;
        filter->new_frame = true;
        filter->cond.notify_one();
        filter->tid.join();
        // re-init backscrub and start thread again
        if (filter->maskctx)
            bs_maskgen_delete(filter->maskctx);
        if (filter->modelname)
            bfree(filter->modelname);
        filter->modelname = model;
        filter->maskctx = bs_maskgen_new(filter->modelname, BS_THREADS, filter->width, filter->height,
            obs_backscrub_dbg, nullptr, nullptr, nullptr, nullptr);
        if (!filter->maskctx) {
            obs_printf(filter, "oops re-initialising backscrub");
            return;
        }
        filter->new_frame = false;
        filter->done = false;
        filter->tid = std::thread(obs_backscrub_mask_thread, filter);
        obs_printf(filter, "update: done");
    }
}
static void obs_backscrub_destroy(void *state) {
    obs_backscrub_filter_t *filter = (obs_backscrub_filter_t *)state;
    obs_printf(filter, "destroy");
    // stop mask thread
    filter->done = true;
    filter->new_frame = true;
    filter->cond.notify_one();
    filter->tid.join();
    // free memory
    if (filter->maskctx)
        bs_maskgen_delete(filter->maskctx);
    if (filter->modelname)
        bfree(filter->modelname);
    delete filter;
    obs_printf(state, "destroy(%p): done");
}
static void obs_backscrub_video_tick(void *state, float secs) { }
static obs_source_frame *obs_backscrub_filter_video(void *state, obs_source_frame *frame) {
    obs_backscrub_filter_t *filter = (obs_backscrub_filter_t *)state;
    // here we do the video frame processing
    // First, map data into an OpenCV Mat object, then convert to BGR24 (default OCV format)
    // for calling libbackscrub
    cv::Mat out;
    switch (frame->format) {
    // TODO: more video formats!
    case VIDEO_FORMAT_YUY2:
    {
        // YUV2 (https://www.fourcc.org/pixel-format/yuv-yuy2/) as OpenCV array is 2x8bit channels
        // in OBS it arrives as a single plane of 16bits/pixel
        cv::Mat obs(frame->height, frame->width, CV_8UC2, frame->data[0], frame->linesize[0]);
        // Re-size to backscrub if required
        if (frame->width != filter->width || frame->height != filter->height)
            cv::resize(obs, obs, cv::Size(filter->width, filter->height));
        // feed the mask thread
        std::lock_guard<std::mutex> hold(filter->lock);
        cv::cvtColor(obs, filter->input, cv::COLOR_YUV2BGR_YUY2);
        filter->new_frame = true;
        filter->cond.notify_one();
        // while we have the lock, grab current mask (if any)
        out = filter->mask.clone();
        break;
    }
    default:
        obs_printf(filter, "filter_video: unsupported frame format: %s", get_video_format_name(frame->format));
        return frame;
    }
    // No mask yet?
    if (out.empty())
        return frame;
    // Re-size back to OBS video if required
    if (frame->width != filter->width || frame->height != filter->height)
        cv::resize(out, out, cv::Size(frame->width, frame->height));
    // Mask the video image, leave the human, green screen the rest
    // blend the edges of the mask.
    for (int row=0; row<out.rows; row+=1) {
        uint8_t *prow = frame->data[0] + frame->linesize[0]*row;
        for (int col=0; col<out.cols; col+=1) {
            int m = (int)(*(out.ptr(row, col)));
            int h = 255-m;
            // blend Y values between human and mask (255)
            prow[2*col] = (uint8_t)((int)(prow[2*col])*h/255+m);
            // blend U/V values between human and zero (green)
            prow[2*col+1] = (uint8_t)((int)(prow[2*col+1])*h/255);
        }
    }
    return frame;
}
static struct obs_source_info backscrub_src {
    // required
    .id = "obs-backscrub",
    .type = OBS_SOURCE_TYPE_FILTER,                  // this displays us in source filter dialog
    .output_flags = OBS_SOURCE_ASYNC_VIDEO,          // this means we only need filter_video()
    .get_name = obs_backscrub_get_name,              // this name is shown in source filter dialog
    .create = obs_backscrub_create,                  // new instance of this filter
    .destroy = obs_backscrub_destroy,                // delete filter instance
    // optional
    .get_defaults = obs_backscrub_get_defaults,      // set default property values
    .get_properties = obs_backscrub_get_properties,  // defines user-adjustable properties (settings)
    .update = obs_backscrub_update,                  // change the filter settings
    .video_tick = obs_backscrub_video_tick,          // frame duration supplied (not absolute time)
    .filter_video = obs_backscrub_filter_video       // process one input to one output frame
};
bool obs_module_load(void) {
    obs_printf(nullptr, "load");
    // here we take the opportunity to ensure dependent components (eg: libbackscrub) are loadable
    obs_register_source(&backscrub_src);
    return true;
}
