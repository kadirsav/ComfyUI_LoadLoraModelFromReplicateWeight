import os

import comfy.sd
import comfy.utils
from huggingface_hub import hf_hub_download
import requests
import hashlib


def download_file(url, filename):
    # make dirs recursively
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")


def get_filename_from_url(url: str, extension: str):
    # md5 hash the url to get a unique filename
    filename = hashlib.md5(url.encode()).hexdigest()
    return f"{filename}.{extension}"


def find_or_create_cache():
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "ComfyUI")):
        cwd = os.path.join(cwd, "ComfyUI")
    if os.path.exists(os.path.join(cwd, "models")):
        cwd = os.path.join(cwd, "models")
    if not os.path.exists(os.path.join(cwd, "huggingface_cache")):
        print("Creating huggingface_cache directory within comfy")
        os.mkdir(os.path.join(cwd, "huggingface_cache"))

    return str(os.path.join(cwd, "huggingface_cache"))


def get_lora_from_url(url: str):
    # clean the url
    url = url.strip()

    # make sure it is a url
    if url is None or url == "":
        raise ValueError("URL is empty")
    if not url.startswith("http"):
        raise ValueError("Invalid URL")

    if "huggingface.co" in url:
        # use hf hub download because it is faster
        # strip the download=true query parameter - HF
        if url.endswith("?download=true"):
            url = url[:-len("?download=true")]

        # handle user copying the blob url - HF
        if '/blob/' in url:
            url = url.replace('/blob/', '/resolve/')

        # https://huggingface.co/owner_name/repo_name/resolve/main/file_name.safetensors

        if not url.lower().endswith("safetensors"):
            raise ValueError("Only safetensors files are supported.")

        # parse hf repo from url
        parser_string = url.replace("http://", "").replace("https://", "")
        parts = parser_string.split("/")
        # ['huggingface.co', 'owner_name', 'repo_name', 'resolve', 'main', 'file_name.safetensors']

        # remove chunks that are not needed
        repo_id = parts[1] + "/" + parts[2]
        filename = parts[-1]

        # extract any depth of subfolders
        # ['huggingface.co', 'owner_name', 'repo_name', 'resolve', 'main', 'subfolder', 'sub-subfolder', 'file_name.safetensors']
        subfolder = None
        subfolder_length = len(parts) - 6

        if subfolder_length > 0:
            subfolder = "/".join(parts[5:-1])

        lora_path = hf_hub_download(
            repo_id=repo_id.strip(),
            subfolder=subfolder,
            filename=filename.strip(),
            cache_dir=find_or_create_cache(),
        )
    elif "civitai.com" in url:
        # good https://civitai.com/api/download/models/0000000000?type=Model&format=SafeTensor
        # bad https://civitai.com/models/111111111111?modelVersionId=0000000000

        # fix bad url
        if "modelVersionId" in url:
            model_version_id = url.split("modelVersionId=")[1]
            model_version_id = model_version_id.split("&")[0]
            url = f"https://civitai.com/api/download/models/{model_version_id}?type=Model&format=SafeTensor"

        if not "SafeTensor" in url:
            raise ValueError("Only safetensors files are supported for security reasons.")

        save_filename = get_filename_from_url(url, "safetensors")

        # check if env var for CIVITAI_API_KEY is set. Add after hashing the url
        if "CIVITAI_API_KEY" in os.environ:
            api_key = os.environ["CIVITAI_API_KEY"]
            if api_key.strip() != "":
                # add the api key to the url
                url += f"&token={api_key}"

        # download the file
        lora_path = os.path.join(find_or_create_cache(), 'civitai', save_filename)
        # check if the file already exists
        if not os.path.exists(lora_path):
            # will fail for most if api key not set
            download_file(url, lora_path)
    else:
        # could be stored somewhere else
        if not url.lower().split("?")[0].endswith("safetensors"):
            raise ValueError("Only safetensors files are supported for security reasons.")

        # try to download the file
        save_filename = get_filename_from_url(url, "safetensors")
        lora_path = os.path.join(find_or_create_cache(), 'general', save_filename)
        # check if the file already exists
        if not os.path.exists(lora_path):
            download_file(url, lora_path)

    return lora_path


class LoadLoraModelOnlyWithUrl:
    CATEGORY = "loaders"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "model": ("MODEL",),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only_from_url"

    def __init__(self):
        self.loaded_lora = None
        self.loaded_lora_path = None

    def load_lora_model_only_from_url(
            self,
            model,
            url: str,
            strength_model: float,
    ):
        if strength_model == 0:
            return (model, )

        lora_path = get_lora_from_url(
            url,
        )
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora_path == lora_path:
                lora = self.loaded_lora
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp
                self.loaded_lora_path = None

        if lora is None:
            lora = comfy.utils.load_torch_file(
                lora_path, safe_load=True)
            self.loaded_lora = lora
            self.loaded_lora_path = lora_path
        model_lora, _ = comfy.sd.load_lora_for_models(
            model,
            None,
            lora,
            strength_model,
            0,
        )
        return (model_lora, )



NODE_CLASS_MAPPINGS = {
    "LoadLoraModelOnlyWithUrl": LoadLoraModelOnlyWithUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraModelOnlyWithUrl": "Load Lora Model Only from URL",
}
