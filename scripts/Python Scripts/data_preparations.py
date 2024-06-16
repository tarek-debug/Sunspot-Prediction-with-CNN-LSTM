# local_script.py

import os
import pandas as pd
import numpy as np
from astropy.io import fits
from skimage.measure import label
import zipfile

# Quantify white spots in images within a folder
def quantify_white_spots_info_in_folder(folder_path):
    results_df = pd.DataFrame(columns=['Image', 'White_Spot_Percentage', 'Num_White_Spots'])

    for filename in os.listdir(folder_path):
        if filename.endswith('.zip'):
            zip_path = os.path.join(folder_path, filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder_path)

            for extracted_filename in os.listdir(folder_path):
                if extracted_filename.endswith('.fits.gz'):
                    fits_path = os.path.join(folder_path, extracted_filename)
                    with fits.open(fits_path) as hdul:
                        image_data = hdul[0].data

                    grayscale_image = np.squeeze(image_data)
                    threshold_value = np.mean(grayscale_image)
                    binary_image = np.where(grayscale_image >= threshold_value, 1, 0)
                    labeled_image = label(binary_image)
                    num_white_spots = np.max(labeled_image)
                    total_pixels = binary_image.size
                    white_spot_percentage = (num_white_spots / total_pixels) * 100

                    new_row = pd.DataFrame({
                        'Image': [extracted_filename],
                        'White_Spot_Percentage': [white_spot_percentage],
                        'Num_White_Spots': [num_white_spots]
                    })
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

            for extracted_filename in os.listdir(folder_path):
                if extracted_filename.endswith('.fits.gz'):
                    extracted_path = os.path.join(folder_path, extracted_filename)
                    os.remove(extracted_path)

    return results_df

def main():
    # Read join.csv
    join_df = pd.read_csv('path/to/join.csv')
    join_df['time'] = pd.to_datetime(join_df['time']).dt.strftime('%Y-%m-%d %H:00:00')
    join_df.set_index('time', inplace=True)

    aggregated_df = pd.DataFrame()

    for year in range(2011, 2019):
        path_to_zip_folder = f'path/to/mask_results/{year}'
        white_spots_info_choronnos_df = quantify_white_spots_info_in_folder(path_to_zip_folder)
        white_spots_info_choronnos_df['time'] = pd.to_datetime(white_spots_info_choronnos_df['Image'].str[:13], format='%Y%m%dT%H').dt.strftime('%Y-%m-%d %H:00:00')
        white_spots_info_choronnos_df.set_index('time', inplace=True)
        merged_df = white_spots_info_choronnos_df.merge(join_df, left_index=True, right_index=True, how='inner')
        aggregated_df = pd.concat([aggregated_df, merged_df])

    output_csv_path = 'path/to/final_2011-2018_merged.csv'
    aggregated_df.to_csv(output_csv_path)

if __name__ == '__main__':
    main()
