import gdown
import pandas as pd

def load_creditcard_data():
    google_path = 'https://drive.google.com/uc?id='
    file_id = '1cA2bkyBdPvNFX8yiL-kyczqlfv8YvjLK'
    output_name = 'train.csv'
    gdown.download(google_path+file_id, output_name, quiet=False)
    return pd.read_csv(output_name)
