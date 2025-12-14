#!/usr/bin/env python3
import argparse
import time
import threading

import cv2
import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord


# Shared latest frames (very simple thread-safe storage)
_lock = threading.Lock()
_latest_rgb = None
_latest_slam = None
_latest_rgb_ts = None
_latest_slam_ts = None


def rgb_callback(image_data: ImageData, image_record: ImageDataRecord):
    global _latest_rgb, _latest_rgb_ts
    img = image_data.to_numpy_array()  # RGB uint8 HxWx3
    with _lock:
        _latest_rgb = img
        _latest_rgb_ts = image_record.capture_timestamp_ns


def slam_callback(image_data: ImageData, image_record: ImageDataRecord):
    global _latest_slam, _latest_slam_ts
    img = image_data.to_numpy_array()  # often grayscale / fisheye depending on profile
    with _lock:
        _latest_slam = img
        _latest_slam_ts = image_record.capture_timestamp_ns


def build_receiver(host: str, port: int, record_to_vrs: str = "") -> receiver.StreamReceiver:
    # HTTP server config
    server_cfg = sdk_gen2.HttpServerConfig()
    server_cfg.address = host
    server_cfg.port = port

    # Typed callback receiver (decoded images)
    sr = receiver.StreamReceiver(enable_image_decoding=True, enable_raw_stream=False)
    sr.set_server_config(server_cfg)

    if record_to_vrs:
        # Many SDKs expect a file path like /path/to/out.vrs
        sr.record_to_vrs(record_to_vrs)

    # Register callbacks
    sr.register_rgb_callback(rgb_callback)
    sr.register_slam_callback(slam_callback)

    return sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=6768, help="Port to bind (default: 6768)")
    parser.add_argument("--record-to-vrs", type=str, default="", help="Optional output .vrs file path")
    parser.add_argument("--show-slam", action="store_true", help="Also show SLAM stream window")
    args = parser.parse_args()

    sr = build_receiver(args.host, args.port, args.record_to_vrs)

    print(f"[Receiver] Starting server on http(s)://{args.host}:{args.port}")
    print("[Receiver] Waiting for device to start streaming...")
    sr.start_server()

    try:
        while True:
            with _lock:
                rgb = None if _latest_rgb is None else _latest_rgb.copy()
                slam = None if _latest_slam is None else _latest_slam.copy()
                rgb_ts = _latest_rgb_ts
                slam_ts = _latest_slam_ts

            # Show RGB
            if rgb is not None:
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    rgb_bgr,
                    f"RGB ts(ns): {rgb_ts}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Aria RGB (press q to quit)", rgb_bgr)

            # Show SLAM (optional)
            if args.show_slam and slam is not None:
                # If slam is HxWx1 or HxW, imshow works directly
                cv2.imshow("Aria SLAM", slam)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Receiver] Quit requested.")
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[Receiver] KeyboardInterrupt, exiting...")

    finally:
        # Some versions may not have explicit stop_server(); safe to just destroy windows
        cv2.destroyAllWindows()
        print("[Receiver] Done.")


if __name__ == "__main__":
    main()
