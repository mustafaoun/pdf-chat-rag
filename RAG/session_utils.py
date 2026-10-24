import hashlib
from pathlib import Path
import tempfile


def build_uploaded_index(content, processor):
    """Give each upload a unique, short-lived file; remove it even on failure."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
            path = Path(temporary.name)
            temporary.write(content)
        return processor(str(path))
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def upload_digest(content):
    return hashlib.sha256(content).hexdigest()
