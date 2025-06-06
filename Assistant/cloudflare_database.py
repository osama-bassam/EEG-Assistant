import threading
from boto3.s3.transfer import TransferConfig
import boto3
import os
import hashlib

# إعدادات R2 من Cloudflare
ACCESS_KEY = 'b4206433e90ba8072ab70d5f38fde091'
SECRET_KEY = '643de7963114823a20c54b59abf297dedf60689132138b8cf5c783b8313a5d7d'
ENDPOINT_URL = 'https://a653512d9901f31c20bcce5bc7b93b7f.r2.cloudflarestorage.com'
BUCKET_NAME = 'eeg-files'

class ProgressPercentage:
    def __init__(self, filename, filesize, filehash, update_progress=None):
        self._filename = filename
        self._filesize = float(filesize)
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._filehash = filehash
        self._update_progress = update_progress
        self._last_printed_percent = -1

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percent = int((self._seen_so_far / self._filesize) * 35)

            if self._update_progress:
                self._update_progress(self._filehash, percent)

            if percent % 5 == 0 and percent != self._last_printed_percent:
                print(f"⬆️ Uploading to R2... {percent}%")
                self._last_printed_percent = percent

def get_file_hash(file_path):
    """Compute the MD5 hash of the file."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_if_file_exists(filehash):
    """Check if the file with the given hash exists in R2."""
    try:
        # Try to get the metadata of the file in the bucket using the hash as the object key
        s3.head_object(Bucket=BUCKET_NAME, Key=f"{filehash}.edf")
        return True  # File exists
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False  # File does not exist
        else:
            raise Exception(f"Error checking file existence: {e}")
        

s3 = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

def upload_edf_to_r2(file_path, update_progress=None):
    """Upload the file to R2 only if it doesn't exist."""
    filehash = get_file_hash(file_path)

    # Check if the file already exists in R2 by its hash
    if check_if_file_exists(filehash):
        print(f"❌ File with hash {filehash} already exists in R2. Skipping upload.")
        return f"File with hash {filehash} already exists. No upload necessary."

    # Proceed with uploading if the file does not exist
    object_name = f"{filehash}.edf"  # Use the hash as the object name to ensure uniqueness
    filesize = os.path.getsize(file_path)

    config = TransferConfig(multipart_threshold=8 * 1024 * 1024)
    try:
        s3.upload_file(
            file_path,
            BUCKET_NAME,
            object_name,
            Config=config,
            Callback=ProgressPercentage(file_path, filesize, filehash, update_progress)
        )
        print(f"✅ Uploaded to R2 as {object_name}")
        return f"Uploaded to R2 as {object_name}"
    except Exception as e:
        print(f"❌ Error uploading to R2: {e}")
        return None
    
def download_edf_from_r2(filehash, update_progress=None):
    s3 = boto3.client(
        's3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    local_path = os.path.join('temp_edf', f"{filehash}.edf")
    os.makedirs('temp_edf', exist_ok=True)

    try:
        print(f"📥 Starting download from R2: {filehash}.edf")

        # نحصل على حجم الملف أولًا
        metadata = s3.head_object(Bucket=BUCKET_NAME, Key=f"{filehash}.edf")
        total_size = metadata['ContentLength']

        with open(local_path, 'wb') as f:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=f"{filehash}.edf")
            progress = DownloadProgress(filehash, total_size, update_progress)
            chunk_size = 8192
            while True:
                chunk = response['Body'].read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                progress(len(chunk))

        print(f"✅ Download complete: {local_path}")
        return local_path

    except Exception as e:
        print(f"❌ Error downloading from R2: {e}")
        return None


class DownloadProgress:
    def __init__(self, filehash, total_bytes, update_progress=None):
        self._filehash = filehash
        self._total = total_bytes
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._update_progress = update_progress
        self._last_printed_percent = -1  # Start with an invalid value

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percent = int(35 + (self._seen_so_far / self._total) * 20)

            if self._update_progress:
                self._update_progress(self._filehash, percent)

            # Only print if percent is a new multiple of 5
            if percent % 5 == 0 and percent != self._last_printed_percent:
                print(f"⬇️ Downloading from R2... {percent}%")
                self._last_printed_percent = percent
