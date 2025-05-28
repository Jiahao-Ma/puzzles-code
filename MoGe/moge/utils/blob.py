from typing import IO, Generator, Tuple, Union, overload
from pathlib import Path, PosixPath, PurePosixPath
import io
import os
import re
import requests
import fnmatch

from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient, BlobClient
import requests.adapters
import requests.packages
from urllib3.util.retry import Retry


__all__ = [
    'download_blob', 'upload_blob', 
    'download_blob_with_cache', 
    'open_blob', 'open_blob_with_cache', 
    'blob_file_exists', 
    'AzureBlobPath','SmartPath'
]

DEFAULT_CREDENTIAL = DefaultAzureCredential()

BLOB_CACHE_DIR = './.blobcache'

def download_blob(blob: Union[str, BlobClient]) -> bytes:
    if isinstance(blob, str):
        blob_client = BlobClient.from_blob_url(blob_client)
    else:
        blob_client = blob
    return blob_client.download_blob().read()


def upload_blob(blob: Union[str, BlobClient], data: Union[str, bytes]):
    if isinstance(blob, str):
        blob_client = BlobClient.from_blob_url(blob)
    else:
        blob_client = blob
    blob_client.upload_blob(data, overwrite=True)


def download_blob_with_cache(container: Union[str, ContainerClient], blob_name: str, cache_dir: str = 'blobcache') -> bytes:
    """
    Download a blob file from a container and return its content as bytes.
    If the file is already present in the cache, it is read from there.
    """
    cache_path = Path(cache_dir) / blob_name
    if cache_path.exists():
        return cache_path.read_bytes()
    data = download_blob(container, blob_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def open_blob(container: Union[str, ContainerClient], blob_name: str) -> io.BytesIO:
    """
    Open a blob file for reading from a container and return its content as a BytesIO object.
    """
    return io.BytesIO(download_blob(container, blob_name))


def open_blob_with_cache(container: Union[str, ContainerClient], blob_name: str, cache_dir: str = 'blobcache') -> io.BytesIO:
    """
    Open a blob file for reading from a container and return its content as a BytesIO object.
    If the file is already present in the cache, it is read from there.
    """
    return io.BytesIO(download_blob_with_cache(container, blob_name, cache_dir=cache_dir))


def blob_file_exists(container: Union[str, ContainerClient], blob_name: str) -> bool:
    """
    Check if a blob file exists in a container.
    """
    if isinstance(container, str):
        container = ContainerClient.from_container_url(container)
    blob_client = container.get_blob_client(blob_name)
    return blob_client.exists()

def is_blob_url(url: str) -> bool:
    return re.match(r'https://[^/]+blob.core.windows.net/+', url) is not None


def split_blob_url(url: str) -> Tuple[str, str, str]:
    match = re.match(r'(https://[^/]+blob.core.windows.net/[^/?]+)(/([^\?]*))?(\?.+)?', url)
    if match:
        container, _, path, sas = match.groups()
        return container, path or '', sas or ''
    raise ValueError(f'Not a valid blob URL: {url}')


def join_blob_path(url: str, *others: str) -> str:
    container, path, sas = split_blob_url(url)
    return container + '/' + os.path.join(path, *others) + sas


class AzureBlobStringWriter(io.StringIO):
    def __init__(self, blob_client: BlobClient, encoding: str = 'utf-8', **kwargs):
        self._encoding = encoding
        self.blob_client = blob_client
        self.kwargs = kwargs
        super().__init__()

    def close(self):
        self.blob_client.upload_blob(self.getvalue().encode(self._encoding), blob_type='BlockBlob', overwrite=True, **self.kwargs)


class AzureBlobBytesWriter(io.BytesIO):
    def __init__(self, blob_client: BlobClient, **kwargs):
        super().__init__()
        self.blob_client = blob_client
        self.kwargs = kwargs

    def close(self):
        self.blob_client.upload_blob(self.getvalue(), blob_type='BlockBlob', overwrite=True, **self.kwargs)


def open_azure_blob(blob: Union[str, BlobClient], mode: str = 'r', encoding: str = 'utf-8', newline: str = None, cache_blob: bool = False, **kwargs) -> IO:
    if isinstance(blob, str):
        blob_client = BlobClient.from_blob_url(blob)
    elif isinstance(blob, BlobClient):
        blob_client = blob
    else:
        raise ValueError(f'Must be a blob URL or a BlobClient object: {blob}')
    
    if cache_blob:
        cache_path = Path(BLOB_CACHE_DIR, blob_client.account_name, blob_client.container_name, blob_client.blob_name)

    if mode == 'r' or mode == 'rb':
        if cache_blob:
            if cache_path.exists():
                data = cache_path.read_bytes()
            else:
                data = blob_client.download_blob(**kwargs).read()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)
        else:
            data = blob_client.download_blob(**kwargs).read()
        if mode == 'r':
            return io.StringIO(data.decode(encoding), newline=newline) 
        else:
            return io.BytesIO(data)
    elif mode == 'w':
        return AzureBlobStringWriter(blob_client, **kwargs)
    elif mode == 'wb':
        return AzureBlobBytesWriter(blob_client, **kwargs)
    else:
        raise ValueError(f'Unsupported mode: {mode}')


def smart_open(path_or_url: Union[Path, str], mode: str = 'r', encoding: str = 'utf-8') -> IO:
    if is_blob_url(str(path_or_url)):
        return open_azure_blob(str(path_or_url), mode, encoding)
    return open(path_or_url, mode, encoding)


class AzureBlobPath(PurePosixPath):
    """
    Implementation of pathlib.Path like interface for Azure Blob Storage.
    """
    container_client: ContainerClient
    _parse_path = PurePosixPath._parse_args if hasattr(PurePosixPath, '_parse_args') else PurePosixPath._parse_path
    
    def __new__(cls, *args, **kwargs):
        """Override the old __new__ method. Parts are parsed in __init__"""
        return object.__new__(cls)

    def __init__(self, root: Union[str, 'AzureBlobPath', ContainerClient], *others: Union[str, PurePosixPath], pool_maxsize: int = 256, retries: int = 3):
        if isinstance(root, AzureBlobPath):
            self.container_client = root.container_client
            parts = root.parts + others
        elif isinstance(root, str):
            url = root
            container, path, sas = split_blob_url(url)
            session = self._get_session(pool_maxsize=pool_maxsize, retries=retries)
            if sas:
                self.container_client = ContainerClient.from_container_url(container + sas, session=session)
            else:
                self.container_client = ContainerClient.from_container_url(container, credential=DEFAULT_CREDENTIAL, session=session)
            parts = (path, *others)
        elif isinstance(root, ContainerClient):
            self.container_client = root
            parts = others
        else:
            raise ValueError(f'Invalid root: {root}')
        
        if hasattr(PurePosixPath, '_parse_args'):
            # For compatibility with Python 3.10
            drv, root, parts = PurePosixPath._parse_args(parts)
            self._drv = drv
            self._root = root
            self._parts = parts
        else:
            super().__init__(*parts)
    
    def _get_session(self, pool_maxsize: int = 1024, retries: int = 3) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE"],
            backoff_factor=1,
            raise_on_status=False,
            read=retries,
            connect=retries,
            redirect=retries,
        )
        adapter = requests.adapters.HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _from_parsed_parts(self, drv, root, parts):
        "For compatibility with Python 3.10"
        return AzureBlobPath(self.container_client, drv, root, *parts)

    def with_segments(self, *pathsegments):
        return AzureBlobPath(self.container_client, *pathsegments)
    
    @property
    def path(self) -> str:
        return '/'.join(self.parts)

    @property
    def blob_client(self) -> BlobClient:
        return self.container_client.get_blob_client(self.path)

    @property
    def url(self) -> str:
        if len(self.parts) == 0:
            return self.container_client.url
        return self.container_client.get_blob_client(self.path).url

    @property
    def container_name(self) -> str:
        return self.container_client.container_name

    @property
    def account_name(self) -> str:
        return self.container_client.account_name
    
    def __str__(self):
        return self.url
    
    def __repr__(self):
        return self.url
    
    def open(self, mode: str = 'r', encoding: str = 'utf-8', cache_blob: bool = False, **kwargs) -> IO:
        return open_azure_blob(self.blob_client, mode, encoding, cache_blob=cache_blob, **kwargs)

    def __truediv__(self, other: Union[str, Path]) -> 'AzureBlobPath':
        return self.joinpath(other)

    def mkdir(self, parents: bool = False, exist_ok: bool = False):
        pass

    def iterdir(self) -> Generator['AzureBlobPath', None, None]:
        path = self.path
        if not path.endswith('/'):
            path += '/'
        for item in self.container_client.walk_blobs(self.path):
            yield AzureBlobPath(self.container_client, item.name)
    
    def glob(self, pattern: str) -> Generator['AzureBlobPath', None, None]:
        special_chars = ".^$+{}[]()|/"  
        for char in special_chars:  
            pattern = pattern.replace(char, "\\" + char)  
        pattern = pattern.replace('**', './/.')  
        pattern = pattern.replace('*', '[^/]*')  
        pattern = pattern.replace('.//.', '.*')  
        pattern = "^" + pattern + "$"  
        reg = re.compile(pattern)

        for item in self.container_client.list_blobs(self.path):
            if reg.match(os.path.relpath(item.name, self.path)):
                yield AzureBlobPath(self.container_client, item.name)

    def exists(self) -> bool:
        return self.blob_client.exists()

    def read_bytes(self, cache_blob: bool = False) -> bytes:
        with self.open('rb', cache_blob=cache_blob) as f:
            return f.read()

    def read_text(self, encoding: str = 'utf-8', cache_blob: bool = False) -> str:
        with self.open('r', encoding=encoding, cache_blob=cache_blob) as f:
            return f.read()
    
    def write_bytes(self, data: bytes):
        self.blob_client.upload_blob(data, overwrite=True)
    
    def write_text(self, data: str, encoding: str = 'utf-8'):
        self.blob_client.upload_blob(data.encode(encoding), overwrite=True)

    def unlink(self):
        self.blob_client.delete_blob()

    def new_client(self) -> 'AzureBlobPath':
        return AzureBlobPath(self.container_client.url, self.path)


class SmartPath(Path, AzureBlobPath):
    """
    Supports both local file paths and Azure Blob Storage URLs.
    """
    def __new__(cls, first: Union[Path, str], *others: Union[str, PurePosixPath]) -> Union[Path, AzureBlobPath]:
        if is_blob_url(str(first)):
            return AzureBlobPath(str(first), *others)  
        return Path(first, *others)


  