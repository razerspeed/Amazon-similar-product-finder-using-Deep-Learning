import gdown


def download_from_drive():
    url = 'https://drive.google.com/uc?id=1YdOTBwrNPg21xfjTbTZwT9ZcdPwE2Rw-'
    output = 'model_efficientnet_b3_IMG_SIZE_512_arcface.bin'
    gdown.download(url, output, quiet=False)
