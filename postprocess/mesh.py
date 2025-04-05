from google_face_mesh import *
import os
from tqdm import tqdm

"""
This should be run AFTER the images are cropped.
Calls the RA_out, RegB_out, RB_out are where the original faces are

"""


def draw_mesh(img_path, save_path):
    lm = Landmarks(img_path, save_path) 
    try: 
        lm.draw_face()
    except:
        pass
    #try: 
    #    if lm.draw_face() is not None:
    #        lm.draw_face()
    #except:
    #    pass
    #elif keypoints_v is None:

    
def iterate_dir(path, save_path):
    for f in tqdm(os.listdir(path)):
        if f != '.ipynb_checkpoints':
            draw_mesh(path + '/' + f, save_path)  
    
### MAIN ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="none", help="experiment_name")
    opt = parser.parse_args()

    # Make dirs to store the face meshes
    os.makedirs("experiments/%s/real_B_mesh" % opt.experiment, exist_ok=True)
    os.makedirs("experiments/%s/reg_B_mesh" % opt.experiment, exist_ok=True)
    os.makedirs("experiments/%s/real_A_mesh" % opt.experiment, exist_ok=True)
    
    # Call the original dirs
    B_path = "experiments/%s/real_B" % opt.experiment
    RB_path = "experiments/%s/reg_B" % opt.experiment
    A_path = "experiments/%s/real_A" % opt.experiment
    
    # Save paths
    Bmesh_path = "experiments/%s/real_B_mesh" % opt.experiment
    RBmesh_path = "experiments/%s/reg_B_mesh" % opt.experiment
    Amesh_path = "experiments/%s/real_A_mesh" % opt.experiment
    
    print("Original dirs...")
    print(B_path)
    print(RB_path)
    print(A_path)
    print("Save paths...")
    print(Bmesh_path)
    print(RBmesh_path)
    print(Amesh_path)
    
    print("------------")
    iterate_dir(B_path, Bmesh_path)
    iterate_dir(RB_path, RBmesh_path)
    iterate_dir(A_path, Amesh_path)
    print("Done!")

