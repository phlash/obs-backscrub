// Playing with OBS plugins..
//
//
#include <obs-module.h>
#include <stdio.h>
#include <stdarg.h>

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
    float last_tick;
    // filter settings
    int setting;
};
static int _obs_play_get_setting(obs_data_t *settings) { return (int)obs_data_get_int(settings, DEMO_SETTING); }
static const char *obs_play_get_name(void *type_data) { return "Phlash playing about"; }
static void *obs_play_create(obs_data_t *settings, obs_source_t *source) {
    op_printf("create\n");
    // here we instantiate a new filter, loading all required resources (eg: model file)
    // and setting initial values for filter settings
    auto *filter = new obs_play_filter_t;
    filter->last_tick = 0;
    filter->setting = _obs_play_get_setting(settings);
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
static void obs_play_destroy(void *state) { delete (obs_play_filter_t *)state; }
static void obs_play_video_tick(void *state, float secs) { ((obs_play_filter_t *)state)->last_tick = secs; }
static obs_source_frame *obs_play_filter_video(void *state, obs_source_frame *input) {
    // here we do the video frame processing
    if (getenv("OBSPLAY_VERBOSE")!= NULL)
        op_printf("filter_video@%f\n", ((obs_play_filter_t *)state)->last_tick);
    return input;
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
