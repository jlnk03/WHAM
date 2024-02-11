# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
import torch.nn as nn
from wham_api import WHAM_API


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.wham_model = WHAM_API()

    def predict(
        self,
        image: Path = Input(description="Video to get pose"),
    ) -> tuple:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

        results, tracking_results, slam_results = self.wham_model(image)

        # keypoints2D, fps, duration, video_bytes, impact_ratio = self.pose_predictor.process_motion(image)
        # keypoints3D = self.pose3D_predictor.run_inference(keypoints2D, image, fps)

        # shoulder_l_s, shoulder_r_s, wrist_l_s, wrist_r_s, hip_l_s, hip_r_s, foot_l_s, eye_l_s, eye_r_s, arm_v = keypoints3D

        # out_path = Path('/tmp/motion.mp4')

        # write video
        # with open(out_path, 'wb') as f:
            # f.write(video_bytes)

        # return shoulder_l_s, shoulder_r_s, wrist_l_s, wrist_r_s, hip_l_s, hip_r_s, foot_l_s, eye_l_s, eye_r_s, arm_v, \
                # duration, fps, impact_ratio, \
                # out_path
        return results, tracking_results, slam_results