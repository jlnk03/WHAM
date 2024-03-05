import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict
from typing import Optional, Union

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

# try: 
#     from lib.models.preproc.slam import SLAMModel
#     _run_global = True
# except: 
#     logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
_run_global = False


from cog import BasePredictor, File as CogFile, Input, Path, Path as CogPath
import torch
import torch.nn as nn
from wham_api import WHAM_API
import tempfile


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file('configs/yamls/demo.yaml')

        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower(), self.cfg.FLIP_EVAL)

        # ========= Load WHAM ========= #
        smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        self.smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
        self.network = build_network(self.cfg, self.smpl)
        self.network.eval()


    def run(self,
        video,
        output_pth,
        network):
    
        cap = cv2.VideoCapture(video)
        assert cap.isOpened(), f'Faild to load video file {video}'
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Whether or not estimating motion in global coordinates
        # run_global = run_global and _run_global
        
        # Preprocess
        with torch.no_grad():
            if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
                    osp.exists(osp.join(output_pth, 'slam_results.pth'))):
                
                
                
                # if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
                # else: slam = None
                slam = None
                
                bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
                while (cap.isOpened()):
                    flag, img = cap.read()
                    if not flag: break
                    
                    # 2D detection and tracking
                    self.detector.track(img, fps, length)
                    
                    # SLAM
                    if slam is not None: 
                        slam.track()
                    
                    bar.next()

                tracking_results = self.detector.process(fps)
                
                if slam is not None: 
                    slam_results = slam.process()
                else:
                    slam_results = np.zeros((length, 7))
                    slam_results[:, 3] = 1.0    # Unit quaternion
            
                # Extract image features
                # TODO: Merge this into the previous while loop with an online bbox smoothing.
                tracking_results = self.extractor.run(video, tracking_results)
                logger.info('Complete Data preprocessing!')
                
                # Save the processed data
                joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
                joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
                logger.info(f'Save processed data at {output_pth}')
            
            # If the processed data already exists, load the processed data
            else:
                tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
                slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
                logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
        
        # Build dataset
        dataset = CustomDataset(self.cfg, tracking_results, slam_results, width, height, fps)
        
        # run WHAM
        results = defaultdict(dict)
        
        n_subjs = len(dataset)
        for subj in range(n_subjs):

            with torch.no_grad():
                if self.cfg.FLIP_EVAL:
                    # Forward pass with flipped input
                    flipped_batch = dataset.load_data(subj, True)
                    _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                    flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                    
                    # Forward pass with normal input
                    batch = dataset.load_data(subj)
                    _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                    pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                    
                    # Merge two predictions
                    flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                    pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                    flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                    avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                    avg_pose = avg_pose.reshape(-1, 144)
                    avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                    
                    # Refine trajectory with merged prediction
                    network.pred_pose = avg_pose.view_as(network.pred_pose)
                    network.pred_shape = avg_shape.view_as(network.pred_shape)
                    network.pred_contact = avg_contact.view_as(network.pred_contact)
                    output = network.forward_smpl(**kwargs)
                    pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
                
                else:
                    # data
                    batch = dataset.load_data(subj)
                    _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                    
                    # inference
                    pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
            

            print('wham done')

            # run smplify smoothing
            # smplify = TemporalSMPLify(self.smpl, img_w=width, img_h=height, device=self.cfg.DEVICE)
            # input_keypoints = dataset.tracking_results[_id]['keypoints']
            # pred = smplify.fit(pred, input_keypoints, **kwargs)

            # print('smplify 1 done')
            
            # with torch.no_grad():
            #     network.pred_pose = pred['pose']
            #     network.pred_shape = pred['betas']
            #     network.pred_cam = pred['cam']
            #     output = network.forward_smpl(**kwargs)
            #     pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

            # print('smplify 2 done')
            
            # ========= Store results ========= #
            pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
            pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
            pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
            pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
            pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
            pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
            
            results[_id]['pose'] = pred_pose
            results[_id]['trans'] = pred_trans
            results[_id]['pose_world'] = pred_pose_world
            results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
            results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
            results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
            results[_id]['frame_ids'] = frame_id


            return results, pred['pose'], pred['poses_root_world']


    def predict(
        self,
        video: Path = Input(description="Video to get pose"),
    ) -> tuple:
        """Run a single prediction on the model"""

        path_name = str(video)

        print(path_name)

        # Output folder
        sequence = '.'.join(path_name.split('/')[-1].split('.')[:-1])
        output_pth = osp.join('/out', sequence)
        os.makedirs(output_pth, exist_ok=True)

        results, pose, pose_root = self.run(path_name, output_pth, network=self.network)

        # results, tracking_results, slam_results = wham_model(video)

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
        return results, pose, pose_root
    
