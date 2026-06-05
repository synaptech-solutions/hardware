#!/usr/bin/env bash
set -euo pipefail

DEVICE="$(readlink -f /dev/v4l/by-id/usb-Elgato_Cam_Link_4K_*-video-index0 2>/dev/null || echo /dev/video4)"

if ! [ -e "$DEVICE" ]; then
    echo "Cam Link not found. Plug it in via USB-C and try again." >&2
    exit 1
fi

if fuser "$DEVICE" >/dev/null 2>&1; then
    echo "$DEVICE is busy — another process is using it. Run: pkill -9 mpv" >&2
    exit 1
fi

exec mpv "av://v4l2:${DEVICE}" \
    --demuxer-lavf-o=video_size=1920x1080,framerate=60,input_format=yuyv422 \
    --profile=low-latency \
    --untimed \
    --no-cache \
    --demuxer-thread=no \
    --demuxer-readahead-secs=0 \
    --vd-lavc-threads=1 \
    --vd-lavc-fast \
    --vd-lavc-skiploopfilter=all \
    --video-latency-hacks=yes \
    --vo=gpu-next \
    --gpu-context=wayland \
    --opengl-swapinterval=0 \
    --framedrop=vo \
    --osd-level=0 \
    --fs \
    --title='Drone Feed (low-latency)'
