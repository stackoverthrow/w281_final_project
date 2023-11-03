import os
import pandas as pd
import cv2


def image_dataframe(images_path):
    val_path = f'{images_path}/val'
    train_path = f'{images_path}/train'
    classes = [f for f in os.listdir(val_path) if f != '.DS_Store']
    image_data = []
    for img_class in classes:
        # get validation images
        val_df = pd.DataFrame(os.listdir(f'{val_path}/{img_class}'), columns=['file_name'])
        val_df['class'] = img_class
        val_df['split'] = 'val'
        image_data.append(val_df)
        train_df = pd.DataFrame(os.listdir(f'{train_path}/{img_class}'), columns=['file_name'])
        train_df['class'] = img_class
        train_df['split'] = 'train'
        image_data.append(train_df)

    combined_df = pd.concat(image_data).reset_index(inplace=False, drop=True)
    split_df = pd.DataFrame(combined_df['file_name'].str.split('_').to_list(),
                            columns=['image_number', 'class_number', 'latitude', 'longitude'])

    # reset so class 1 is class 0, this will jive better with soft max outputs
    split_df['class_number'] = pd.to_numeric(split_df['class_number']) - 1
    split_df['latitude'] = pd.to_numeric(split_df['latitude'])
    split_df['longitude'] = list(map(lambda x: x.replace('.png', ''), split_df['longitude']))
    split_df['longitude'] = pd.to_numeric(split_df['longitude'])
    split_df['file_name'] = combined_df['file_name']
    split_df['class'] = combined_df['class']
    split_df['split'] = combined_df['split']
    split_df['full_path'] = split_df.apply(lambda x: f'{images_path}/{x["split"]}/{x["class"]}/{x["file_name"]}', axis=1)
    split_df.to_csv('images.csv')
    return split_df


