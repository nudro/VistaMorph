from landmark_util import * 
from alignment_util import * 
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict 
import argparse
import glob
from natsort import natsorted
import pickle
import numpy as np
import math
from tqdm import tqdm
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *
import pandas as pd
from shutil import copyfile
import shutil

#=======================

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default='train', help="train or test")
opt = parser.parse_args()
print(opt)

class Landmarks(object):

    def __init__(self, img_path):
        self.img_path = img_path
        
    def fetch(self):
        mp_face_mesh = mp.solutions.face_mesh
        # Initialize MediaPipe Face Mesh.
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        image = cv2.imread(self.img_path, flags=cv2.IMREAD_COLOR) #open image
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            score = 1
            landmarks = results.multi_face_landmarks

            # we want the dict values x, y, z into an array
            keypoints = protobuf_to_dict(landmarks[0]) ##iterate for a dictionary <- # Ref: https://github.com/google/mediapipe/issues/1020
            my_keypoints = np.array(list(keypoints.values()))  #grab the values and convert to an array
            
            return score, my_keypoints
        
        else:
            print("None {}".format(self.img_path))
            


def keypoints_formatter(kps):
    keypoints_list = [] #this is the keypoints for this set of landmarks only 
    for i in range(0, kps.size): # (0, 468)
        keypoints_list.append(list(kps[0][i].values())) 
        
    return np.array(keypoints_list)
    
    
def face_mesh1(fname1):
    # Call the Google Mesh Landmarks
    lm1 = Landmarks(fname1)
    score1, keypoints1 = lm1.fetch()
    #lm1.draw_face()
    k1 = keypoints_formatter(keypoints1)
    
    return k1

def face_mesh2(fname2):
    # Call the Google Mesh Landmarks
    lm2 = Landmarks(fname2)
    score2, keypoints2 = lm2.fetch()
    #lm2.draw_face()
    k2 = keypoints_formatter(keypoints2)
    
    return k2


def mse_calc(A_path, al_B):
    files = []
    scores = []

    #Compute MSE(og_A, al_B)
    names = os.listdir(A_path)

    for filename in tqdm(names): 
        A_dir = os.path.join(A_path, filename)
        aligned_thr = os.path.join(al_B, filename) # get the matching B

        if os.path.isfile(A_dir) and os.path.isfile(aligned_thr):
            # half of the original visible A files will not be in thermal B
            try: 
                k1 = face_mesh1(A_dir)
            except:
                print("Could not detect landmarks, selecting Aligned A and Aligned B.")
                scores.append(1.0)
                files.append(filename)
                continue
                
            # some of the thermal faces may be too close to get landmarks
            try: 
                k2 = face_mesh2(aligned_thr)
                #print(mean_squared_error(k1,k2))
                scores.append(mean_squared_error(k1,k2))
                files.append(filename)
            except:
                print("Could not detect landmarks, selecting Aligned A and Aligned B.")
                scores.append(1.0)
                files.append(filename)
                continue
    print(len(files))
    return files, scores



class LOFTR(object):
    
    def __init__(self, A_path, al_B, thresh=5.0):
        self.A_path = A_path
        self.al_B = al_B
        self.thresh = thresh
        
    def load_torch_image(self, fname):
        img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
        img = K.color.bgr_to_rgb(img)
        
        return img
    
    # function to calculate the score based on arctan2(f1, f2)
    def loftr_analysis(self, fname1, fname2):
        img1 = self.load_torch_image(fname1)
        img2 = self.load_torch_image(fname2)
        #print("images loaded")

        matcher = KF.LoFTR(pretrained='outdoor')

        input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.no_grad():
            correspondences = matcher(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()

        H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000) 
        inliers = inliers > 0

        #print("Number of inliers:", len(inliers))
        #print("Fundamental Matrix:", H)

        angles = []
        for i in range(0, len(inliers)):
            dY = mkpts1[i][1] - mkpts0[i][1]
            dX = mkpts1[i][0] - mkpts0[i][0]
            angle = np.degrees(np.arctan2(dY, dX)) 
            angles.append(angle)

        #f = np.where((np.array(angles)<= 50.0) & (np.array(angles) >= -50.0))
        g = np.where(abs(np.array(angles)) <= self.thresh) # 5.0
        score = len(g[0])/len(inliers)

        #print("Perc < 5 deg variant: {:.2f}% ".format(len(g[0])/len(inliers) * 100))
        #print("Average using abs values:", abs(np.array(angles)).mean())

        return score

    
    def arctan2_dir(self):

        files = []
        scores = []

        names = os.listdir(self.al_B) # grab all names in directory
        
        for filename in tqdm(names): 
            B_al_file = os.path.join(self.al_B, filename) 
            A_file = os.path.join(self.A_path, filename) # grab only the matching filenames from A
            
            if os.path.isfile(A_file) and os.path.isfile(B_al_file):
                perc_5 = self.loftr_analysis(A_file, B_al_file) # ? self loftr_analysis(self, fname1, fname2):
                #print(perc_5)
                scores.append(perc_5)
                files.append(filename)
            else:
                #print("Continue")
                continue
            
        #print("number of files:", len(files))
        return files, scores


def make_choices(files_og, scores_og, files_al, scores_al, phase):

    original = {'file':files_og,'score_og':scores_og}
    df_og = pd.DataFrame(original, columns=['file','score_og'])

    aligned = {'file':files_al,'score_al':scores_al}
    df_al = pd.DataFrame(aligned, columns=['file','score_al'])


    """
                                                    file  score_og  score_al
    0     SUBJe747ed03_e_2019-11-13_14-32-23-995.png  0.000000  0.045455
    1     SUBJ860cbd63_e_2019-11-12_14-53-15-238.png  0.000000  0.030303
    2     SUBJ83c39a1f_e_2019-11-12_12-34-37-142.png  0.057143  0.038462
    3     SUBJ7d344372_e_2019-11-15_09-32-28-538.png  0.023810  0.148148
    4     SUBJ39387a7d_e_2019-11-06_14-38-28-790.png  0.054054  0.000000
    ...                                          ...       ...       ...
    1832  SUBJ776771ff_e_2019-11-15_08-47-04-181.png  0.018182  0.000000
    1833  SUBJ26b9c433_e_2019-11-07_12-33-00-842.png  0.000000  0.000000
    1834  SUBJb7c93da8_e_2019-11-14_16-37-31-218.png  0.055556  0.000000
    1835  SUBJ681fce8d_e_2019-11-14_12-36-34-080.png  0.089286  0.000000
    1836  SUBJa1a88fab_e_2019-11-04_14-06-02-984.png  0.000000  0.000000

    """

    df = pd.merge(df_og, df_al, how="left", on="file")
    df.to_csv('/home/local/AD/cordun1/DEVCOM/dataset/{}_loftr_scores.csv'.format(phase))
    print(df)


    # Official ones
    og_src = '/home/local/AD/cordun1/DEVCOM/dataset/{}/A/og/'.format(phase)
    al_src = '/home/local/AD/cordun1/DEVCOM/dataset/{}/A/al/'.format(phase)
    target = '/home/local/AD/cordun1/DEVCOM/dataset/{}_selections/A/'.format(phase)
    # B_al is already in test_selections

    for i in tqdm(range(0, len(df))):
        if df.iloc[i]['score_og'] > df.iloc[i]['score_al']: # if OG better > AL
            shutil.copy(og_src + df.iloc[i]['file'], target) #
            #print("original file:", df.iloc[i]['file'])
        elif df.iloc[i]['score_og'] < df.iloc[i]['score_al']:
            shutil.copy(al_src + df.iloc[i]['file'], target) 
            #print("aigned file:", df.iloc[i]['file'])
        elif df.iloc[i]['score_og'] == df.iloc[i]['score_al']: # tie or both are zero, choose the Aligned A
            shutil.copy(al_src + df.iloc[i]['file'], target) 
            #print("tie file:", df.iloc[i]['file'])
    print("Done!")
    

if __name__ == '__main__':
    

    """ Test Set Paths"""
    # These will be the src paths from which you will mvoe to the target 
    og_A = '/home/local/AD/cordun1/DEVCOM/dataset/{}/A/og/'.format(opt.phase)
    al_A = '/home/local/AD/cordun1/DEVCOM/dataset/{}/A/al/'.format(opt.phase) #aligned
    #og_B = './dataset/{}/B/og/'.format(opt.mode) # regular 
    #al_B = '/home/local/AD/cordun1/DEVCOM/dataset/{}/B/al/'.format(opt.phase) # aligned
    
    # For training set use this: b/c the aligned B's are already in the folder
    # I could have used it for the test set, too.
    al_B = '/home/local/AD/cordun1/DEVCOM/dataset/{}_selections/B/'.format(opt.phase) # I already made this directory
    
    print("...Original A and Aligned B...")
    tool1 = LOFTR(og_A, al_B, thresh=5.0)
    files_og, scores_og = tool1.arctan2_dir()
    print("Done!")
    
    print("...Aligned A and Aligned B...")
    tool2 = LOFTR(al_A, al_B, thresh=5.0)
    files_al, scores_al = tool2.arctan2_dir()
    print("Done!")
    
    make_choices(files_og, scores_og, files_al, scores_al, opt.phase)
    

    

    

                    
                

                
