import numpy as np
import cv2
import time
import sys
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

try:
    from camera_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        print("No cameras detected!")
        sys.exit()

    with sdk.open_camera(available_cameras[0]) as camera:
        print(f"Opened camera: {camera.model} (SN: {camera.serial_number})")

        # ---------------------------------------------------------------------
        # 1) Set the ROI (region of interest) to 1/2 sensor: 476 x 432, centered
        #
        #    Full sensor: 1440 x 1080
        #    Desired ROI: 144 wide, 108 high
        #    Offsets to center: (x_off, y_off) = (648, 486)
        # ---------------------------------------------------------------------
        camera.roi_width_pixels  = 476
        camera.roi_height_pixels = 432
        camera.roi_x_pixels      = 568
        camera.roi_y_pixels      = 224

        # ---------------------------------------------------------------------
        # 2) Set up desired camera parameters (exposure, trigger, etc.)
        # ---------------------------------------------------------------------
        camera.exposure_time_us = 1            # e.g. 1ms exposure
        camera.frames_per_trigger_zero_for_unlimited = 0   # continuous mode
        camera.image_poll_timeout_ms = 0       # 1 second polling
        camera.frame_rate_control_value = 80     # example: limit frame rate
        camera.is_frame_rate_control_enabled = True

        # Optionally set the operation mode:
        # camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGER
        # (If you need software triggers. Otherwise default is free-run.)

        # ---------------------------------------------------------------------
        # 3) Arm the camera and start acquiring
        # ---------------------------------------------------------------------
        camera.arm(2)  # '2' typically means "frame‐trigger mode = continuous"
        camera.issue_software_trigger()  # Only needed if using software triggers
        frame_count_for_save = 100  # or any frame index you want to capture
        save_done = False           # flag so we only save once

        try:
            while True:
                frame = camera.get_pending_frame_or_null()
                if frame is not None:
                    # Copy the raw image data
                    image_buffer_copy = np.copy(frame.image_buffer)

                    numpy_shaped_image = image_buffer_copy.reshape(
                        camera.image_height_pixels,
                        camera.image_width_pixels
                    )
                    nd_image_array = np.dstack([numpy_shaped_image]*3).astype(np.uint8)

                    # Save exactly once at a chosen frame count
                    if (frame.frame_count == frame_count_for_save) and (not save_done):
                        cv2.imwrite("frame_100.png", nd_image_array)
                        print("Saved frame_100.png")
                        save_done = True

                    cv2.imshow("TSI Camera (ROI 1/2 sensor)", nd_image_array)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("No frame returned this cycle.")
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("Streaming loop interrupted by user.")

        # Clean up
        cv2.destroyAllWindows()
        camera.disarm()

print("Program completed.")
