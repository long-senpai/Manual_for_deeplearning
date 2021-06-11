import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import shutil

path = "/mnt/642C9F7E0555E58A/Nobi/fix_Data/data_nobi_invalid_human_accept"
def check_valid_csv(csv, folder):
    df = pd.read_csv(csv)
    for fn in tqdm(glob(f"{folder}/*.jpg")):
        #
        _id = os.path.basename(fn)[:-6]
        # print(_id)

        index = df.loc[:, "old_image_id"] == _id  # +".jpg"
        room = (df.loc[index, "room_type"])
        if len(df.loc[index, "old_image_id"]) == 0:
            pass
        else:

            if room.values[0] == "living":
               # import ipdb; ipdb.set_trace()
                with open(fn.replace(".jpg", ".txt"), "r") as f:
                    lines = f.read().splitlines()
                with open(fn.replace(".jpg", ".txt"), "w") as f:
                    temp_list = []
                    for line in lines:
                        if line[0] != "6" and line[0] != "5" :
                            temp_list.append(line)
                    f.write("\n".join(temp_list))
             
check_valid_csv("export_long_with_old_id.csv",
                folder=path)

