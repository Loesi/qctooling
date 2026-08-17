from pydantic import BaseModel, PrivateAttr
from typing import Union, Generator, Set
from contextlib import contextmanager
import logging, pathlib, shutil, tempfile, uuid

# Filesystems that require round trips over the network. On these, parsing
# line-by-line (or many small reads) is much slower than on local disks, so
# files living on them are first copied to the local temp dir.
_NETWORK_FSTYPES: Set[str] = {
    "9p",
    "cifs", "smb2", "smb3", "smbfs",
    "nfs", "nfs4",
    "sshfs", "fuse.sshfs",
    "davfs", "davfs2",
    "gvfs", "gvfs-fuse", "gvfsd-fuse",
    "ceph", "fuse.ceph", "fuse.s3fs", "fuse.rclone",
    "lustre", "gpfs", "mmfs", "beegfs", "fhgfs",
    "glusterfs", "pvfs2", "ocfs2",
    "curlftpfs", "fuse.portal",
}


def _fstype_for(path: pathlib.Path) -> str:
    """Return the filesystem type of `path` (from ``/proc/mounts``).

    The mount point with the longest prefix match is used. Returns "" when
    the fstype cannot be determined (non-Linux or unreadable mounts table).
    """
    mount_info = pathlib.Path(path).resolve()
    best, best_type = "", ""
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if mount_info.is_relative_to(mnt) and len(mnt) > len(best):
                    best, best_type = mnt, parts[2]
    except OSError:
        return ""
    return best_type


def _is_network_path(path: pathlib.Path) -> bool:
    return _fstype_for(path) in _NETWORK_FSTYPES


class ParserBase(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    path: pathlib.Path
    baseName: str
    lvprt: int = logging.WARNING

    _logger: logging.Logger = PrivateAttr()

    def model_post_init(self, __context) -> None:
        # unique per instance -> independent level control
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}.{self.baseName}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(self.lvprt)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def _cp_to_tmp(self, filepath: pathlib.Path) -> pathlib.Path:
        """Copy a file into the system temp dir and return the temp path."""
        filepath = pathlib.Path(filepath)
        tmp_path = (
            pathlib.Path(tempfile.gettempdir())
            / f"{filepath.stem}-{uuid.uuid4().hex}{filepath.suffix}"
        )
        shutil.copy2(filepath, tmp_path)
        self._logger.debug("copied '%s' -> '%s'", filepath, tmp_path)
        return tmp_path

    def _delete_tmp(self, tmp_path: pathlib.Path) -> None:
        """Remove a temp copy created by ``_cp_to_tmp`` (no-op if gone)."""
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        self._logger.debug("removed temp copy '%s'", tmp_path)

    @contextmanager
    def _local_path(self, filepath: pathlib.Path) -> Generator[pathlib.Path]:
        """Yield a path that is local to this machine.

        If ``filepath`` lives on a network filesystem it is first copied into
        the system temp dir and that copy is removed again when the block ends.
        Local files are yielded untouched.
        """
        filepath = pathlib.Path(filepath)
        if not _is_network_path(filepath):
            yield filepath
            return
        tmp_path = self._cp_to_tmp(filepath)
        try:
            yield tmp_path
        finally:
            self._delete_tmp(tmp_path)
