"""File Utils module."""
import logging
import os
import multiprocessing
import subprocess
import time
import fsspec
import torch
from tqdm import tqdm


def remote_sync_s3(local_dir, remote_dir):
    """Remote Sync S3.

        Args:
            local_dir: local_dir parameter.
            remote_dir: remote_dir parameter.
        """
    # skip epoch_latest which can change during sync.
    result = subprocess.run(["aws", "s3", "sync", local_dir, remote_dir, '--exclude',
                            '*epoch_latest.pt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False

    logging.info(f"Successfully synced with S3 bucket")
    return True


def remote_sync_fsspec(local_dir, remote_dir):
    """Remote Sync Fsspec.

        Args:
            local_dir: local_dir parameter.
            remote_dir: remote_dir parameter.
        """
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if 'epoch_latest.pt' in k:
            continue

        logging.info(f'Attempting to sync {k}')
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f'Skipping remote sync for {k}.')
            continue

        try:
            logging.info(f'Successful sync for {k}.')
            b[k] = a[k]
        except Exception as e:
            logging.info(f'Error during remote sync for {k}: {e}')
            return False

    return True


def remote_sync(local_dir, remote_dir, protocol):
    """Remote Sync.

        Args:
            local_dir: local_dir parameter.
            remote_dir: remote_dir parameter.
            protocol: protocol parameter.
        """
    logging.info('Starting remote sync.')
    if protocol == 's3':
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == 'fsspec':
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error('Remote protocol not known')
        return False


def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    """Keep Running Remote Sync.

        Args:
            sync_every: sync_every parameter.
            local_dir: local_dir parameter.
            remote_dir: remote_dir parameter.
            protocol: protocol parameter.
        """
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)


def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    """Start Sync Process.

        Args:
            sync_every: sync_every parameter.
            local_dir: local_dir parameter.
            remote_dir: remote_dir parameter.
            protocol: protocol parameter.
        """
    p = multiprocessing.Process(target=keep_running_remote_sync, args=(sync_every, local_dir, remote_dir, protocol))
    return p

# Note: we are not currently using this save function.


def pt_save(pt_obj, file_path):
    """Pt Save.

        Args:
            pt_obj: pt_obj parameter.
            file_path: file_path parameter.
        """
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)


def pt_load(file_path, map_location=None):
    """Pt Load.

        Args:
            file_path: file_path parameter.
            map_location: map_location parameter.
        """
    if file_path.startswith('s3'):
        logging.info('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def check_exists(file_path):
    """Check Exists.

        Args:
            file_path: file_path parameter.
        """
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True
