import requests
import zipfile
import os
import sys
import re
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    """
    Downloads a file from Google Drive using the file ID
    with proper handling of large files
    """
    URL = "https://drive.google.com/uc"
    
    session = requests.Session()
    response = session.get(URL, params={'id': id, 'export': 'download'}, stream=True)
    
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            confirm_token = value
            break
    
    if confirm_token:
        params = {'id': id, 'export': 'download', 'confirm': confirm_token}
    else:
        params = {'id': id, 'export': 'download'}
    
    response = session.get(URL, params=params, stream=True)
    
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        direct_link_match = re.search(r'href="(\/uc\?export=download[^"]+)"', response.text)
        if direct_link_match:
            direct_link = 'https://drive.google.com' + direct_link_match.group(1).replace('&amp;', '&')
            print(f"Using direct download link: {direct_link}")
            response = session.get(direct_link, stream=True)
        else:
            print("Failed to extract direct download link.")
            print("Please download the file manually from the Google Drive link.")
            print("After downloading, place it in the current directory and run:")
            print(f"extract_zip('{destination}', 'finetuned_bert_sentiment_model')")
            return False
    
    file_size = int(response.headers.get('content-length', 0))
    
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=destination)
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                f.write(chunk)
    
    progress.close()
    
    try:
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            pass
        return True
    except zipfile.BadZipFile:
        print(f"Downloaded file is not a valid ZIP file.")
        print("Google Drive might have returned an HTML page instead of the file.")
        print("Please download the file manually from the Google Drive link.")
        os.remove(destination)
        return False

def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to the specified directory
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            progress = tqdm(total=len(file_list), desc="Extracting files")
            
            for file in file_list:
                zip_ref.extract(file, extract_to)
                progress.update(1)
            
            progress.close()
        
        print("Extraction complete!")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
        return False

def use_gdown_if_available(file_id, destination):
    """
    Try to use gdown if it's available, as it handles Google Drive downloads better
    """
    try:
        import gdown
        print("Using gdown to download file...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        return True
    except ImportError:
        print("gdown not installed. Falling back to requests.")
        return False
    except Exception as e:
        print(f"Error using gdown: {e}")
        print("Falling back to requests.")
        return False

def main():
    extract_only = False
    if len(sys.argv) > 1 and sys.argv[1] == '--extract-only':
        extract_only = True
    
    file_id = "1j8Z7f-5BWTCzQBvXCSK6VZajsWFIVrn9"
    
    zip_filename = "finetuned_bert_sentiment_model.zip"
    extract_folder = os.getcwd()
    
    if extract_only:
        if os.path.exists(zip_filename):
            print(f"Extracting {zip_filename} without downloading...")
            if extract_zip(zip_filename, extract_folder):
                print(f"Successfully extracted to {extract_folder}")
            else:
                print("Extraction failed")
        else:
            print(f"Cannot extract: {zip_filename} does not exist")
        return
    
    success = False
    
    try:
        import gdown
        print("gdown is installed, trying to use it for download...")
        success = use_gdown_if_available(file_id, zip_filename)
    except ImportError:
        print("gdown not installed, trying to install it...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "gdown"])
            print("gdown installed successfully, trying to use it...")
            success = use_gdown_if_available(file_id, zip_filename)
        except Exception as e:
            print(f"Could not install gdown: {e}")
            print("Falling back to custom download method...")
            success = download_file_from_google_drive(file_id, zip_filename)
    
    if not success:
        print("Could not download with gdown, falling back to custom method...")
        success = download_file_from_google_drive(file_id, zip_filename)
    
    if success and os.path.exists(zip_filename):
        print(f"Successfully downloaded {zip_filename}")
        
        if extract_zip(zip_filename, extract_folder):
            print(f"Successfully extracted to {extract_folder}")
            
            # remove the zip file after extraction
            # os.remove(zip_filename)
            # print(f"Deleted {zip_filename}")
        else:
            print("Extraction failed")
    else:
        print("Download failed")
        print("\nAlternative download instructions:")
        print("1. Open the Google Drive link in your browser")
        print("2. Click the Download button in the top right")
        print("3. Save the file as 'finetuned_bert_sentiment_model.zip'")
        print("4. Place it in the current directory")
        print("5. Run this script again to extract it")

if __name__ == "__main__":
    main()