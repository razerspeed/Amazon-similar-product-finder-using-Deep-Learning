import gdown


def download_from_drive():
    print("downloading model")
    url = 'https://drive.google.com/uc?id=1Xc9te-Q8teJHrE6omfa0YNxP0l5TgBmm'
    output = 'model_efficientnet_b3_IMG_SIZE_512_arcface.bin'
    gdown.download(url, output, quiet=False)

