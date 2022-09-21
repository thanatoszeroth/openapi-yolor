import gdown

# a folder
url = "https://drive.google.com/drive/folders/19l8NvZEzu_PN8rAH0zZaDBz4GAuK95qI?usp=sharing"
gdown.download_folder(url, quiet=True, use_cookies=False)