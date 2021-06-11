from wavesim_params import wavesim_duration


emitter_beep_sweep_len = wavesim_duration // 16
emitter_sequential_delay = wavesim_duration // 8


emitter_format_beep = "beep"
emitter_format_sweep = "sweep"
emitter_format_impulse = "impulse"
emitter_format_all = [emitter_format_beep, emitter_format_sweep, emitter_format_impulse]


emitter_arrangement_mono = "mono"
emitter_arrangement_stereo = "stereo"
emitter_arrangement_surround = "surround"
emitter_arrangement_all = [
    emitter_arrangement_mono,
    emitter_arrangement_stereo,
    emitter_arrangement_surround,
]


receiver_arrangement_flat = "flat"
receiver_arrangement_grid = "grid"
receiver_arrangement_all = [receiver_arrangement_flat, receiver_arrangement_grid]


input_format_audioraw = "audioraw"
input_format_audiowaveshaped = "audiowaveshaped"
input_format_spectrogram = "spectrogram"
input_format_gcc = "gcc"
input_format_gccphat = "gccphat"
input_format_all = [
    input_format_audioraw,
    input_format_audiowaveshaped,
    input_format_spectrogram,
    input_format_gcc,
    input_format_gccphat,
]

output_format_depthmap = "depthmap"
output_format_heatmap = "heatmap"
output_format_sdf = "sdf"
output_format_all = [output_format_depthmap, output_format_heatmap, output_format_sdf]
