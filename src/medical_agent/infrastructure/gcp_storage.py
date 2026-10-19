"""
GCP Cloud Storage Client for Medical Paper Storage

Provides operations for uploading, downloading, and managing
medical research PDFs in Google Cloud Storage.
"""

import logging
from pathlib import Path
from typing import BinaryIO, cast

from google.cloud import storage  # type: ignore[attr-defined]
from google.cloud.exceptions import NotFound
from tenacity import retry, stop_after_attempt, wait_exponential

from medical_agent.core.config import settings
from medical_agent.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class GCPStorageClient:
    """
    Client for interacting with GCP Cloud Storage.

    Handles PDF upload/download operations for medical research papers.
    """

    def __init__(
        self,
        project_id: str | None = None,
        bucket_name: str | None = None,
        credentials_path: str | None = None,
    ):
        """
        Initialize the GCP Storage client.

        Args:
            project_id: GCP project ID (defaults to settings)
            bucket_name: Storage bucket name (defaults to settings)
            credentials_path: Path to service account JSON key
        """
        self.project_id = project_id or settings.gcp_project_id
        self.bucket_name = bucket_name or settings.gcp_bucket_name
        self.credentials_path = credentials_path or settings.gcp_credentials_path

        self._client: storage.Client | None = None
        self._bucket: storage.Bucket | None = None

    @property
    def client(self) -> storage.Client:
        """Get or create the storage client."""
        if self._client is None:
            if self.credentials_path:
                self._client = storage.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project_id,
                )
            else:
                # Uses default credentials (ADC) - works in Cloud Run
                self._client = storage.Client(project=self.project_id)
        return self._client

    @property
    def bucket(self) -> storage.Bucket:
        """Get or create the bucket reference."""
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def is_configured(self) -> bool:
        """Check if GCP storage is properly configured."""
        return bool(self.project_id and self.bucket_name)

    def verify_connection(self) -> bool:
        """
        Verify that we can connect to the bucket.

        Returns:
            True if connection is successful

        Raises:
            StorageError: If connection fails
        """
        if not self.is_configured():
            raise StorageError("GCP Storage is not configured")

        try:
            # Check if bucket exists by trying to get its metadata
            self.bucket.reload()
            logger.info(f"Successfully connected to GCP bucket: {self.bucket_name}")
            return True
        except NotFound as err:
            raise StorageError(f"Bucket not found: {self.bucket_name}") from err
        except Exception as e:
            raise StorageError(f"Failed to connect to GCP Storage: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def upload_pdf(
        self,
        file_data: bytes | BinaryIO,
        destination_path: str,
        content_type: str = "application/pdf",
        metadata: dict | None = None,
    ) -> str:
        """
        Upload a PDF file to the bucket.

        Args:
            file_data: PDF content as bytes or file-like object
            destination_path: Path within the bucket (e.g., "papers/2024/study.pdf")
            content_type: MIME type of the file
            metadata: Optional custom metadata to attach

        Returns:
            The GCS URI of the uploaded file (gs://bucket/path)

        Raises:
            StorageError: If upload fails
        """
        if not self.is_configured():
            raise StorageError("GCP Storage is not configured")

        try:
            blob = self.bucket.blob(destination_path)

            # Set metadata if provided
            if metadata:
                blob.metadata = metadata

            # Upload based on input type
            if isinstance(file_data, bytes):
                blob.upload_from_string(file_data, content_type=content_type)
            else:
                blob.upload_from_file(file_data, content_type=content_type)

            gcs_uri = f"gs://{self.bucket_name}/{destination_path}"
            logger.info(f"Uploaded PDF to {gcs_uri}")
            return gcs_uri

        except Exception as e:
            logger.error(f"Failed to upload PDF: {e}")
            raise StorageError(f"Failed to upload PDF: {e}") from e

    def upload_pdf_from_path(
        self,
        local_path: str | Path,
        destination_path: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Upload a PDF file from local filesystem.

        Args:
            local_path: Path to the local PDF file
            destination_path: Path within bucket (defaults to filename)
            metadata: Optional custom metadata

        Returns:
            The GCS URI of the uploaded file
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise StorageError(f"File not found: {local_path}")

        if destination_path is None:
            destination_path = f"papers/{local_path.name}"

        with open(local_path, "rb") as f:
            return self.upload_pdf(f, destination_path, metadata=metadata)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def download_pdf(self, source_path: str) -> bytes:
        """
        Download a PDF file from the bucket.

        Args:
            source_path: Path within the bucket

        Returns:
            PDF content as bytes

        Raises:
            StorageError: If download fails
        """
        if not self.is_configured():
            raise StorageError("GCP Storage is not configured")

        try:
            blob = self.bucket.blob(source_path)
            content = cast(bytes, blob.download_as_bytes())
            logger.info(f"Downloaded PDF from {source_path}")
            return content
        except NotFound as err:
            raise StorageError(f"PDF not found: {source_path}") from err
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            raise StorageError(f"Failed to download PDF: {e}") from e

    def download_pdf_to_path(
        self,
        source_path: str,
        local_path: str | Path,
    ) -> Path:
        """
        Download a PDF file to local filesystem.

        Args:
            source_path: Path within the bucket
            local_path: Local destination path

        Returns:
            Path to the downloaded file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.download_pdf(source_path)
        local_path.write_bytes(content)

        logger.info(f"Saved PDF to {local_path}")
        return local_path

    def delete_pdf(self, path: str) -> bool:
        """
        Delete a PDF file from the bucket.

        Args:
            path: Path within the bucket

        Returns:
            True if deleted successfully
        """
        if not self.is_configured():
            raise StorageError("GCP Storage is not configured")

        try:
            blob = self.bucket.blob(path)
            blob.delete()
            logger.info(f"Deleted PDF: {path}")
            return True
        except NotFound:
            logger.warning(f"PDF not found for deletion: {path}")
            return False
        except Exception as e:
            raise StorageError(f"Failed to delete PDF: {e}") from e

    def list_pdfs(self, prefix: str = "") -> list[str]:
        """
        List all PDFs in the bucket with a given prefix.

        Args:
            prefix: Path prefix to filter (default: "")

        Returns:
            List of PDF filenames
        """
        if not self.is_configured():
            raise StorageError("GCP Storage is not configured")

        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

            files = []
            for blob in blobs:
                if blob.name.endswith(".pdf"):
                    files.append(blob.name)

            return files
        except Exception as e:
            raise StorageError(f"Failed to list PDFs: {e}") from e

    def get_signed_url(
        self,
        path: str,
        expiration_minutes: int = 60,
    ) -> str:
        """
        Generate a signed URL for temporary access to a PDF.

        Args:
            path: Path within the bucket
            expiration_minutes: URL validity period

        Returns:
            Signed URL string
        """
        if not self.is_configured():
            raise StorageError("GCP Storage is not configured")

        from datetime import timedelta

        try:
            blob = self.bucket.blob(path)
            url = cast(
                str,
                blob.generate_signed_url(
                    expiration=timedelta(minutes=expiration_minutes),
                    method="GET",
                ),
            )
            return url
        except Exception as e:
            raise StorageError(f"Failed to generate signed URL: {e}") from e

    def pdf_exists(self, path: str) -> bool:
        """Check if a PDF exists in the bucket."""
        if not self.is_configured():
            return False

        try:
            blob = self.bucket.blob(path)
            return cast(bool, blob.exists())
        except Exception:
            return False


# Global client instance
_storage_client: GCPStorageClient | None = None


def get_storage_client() -> GCPStorageClient:
    """Get or create the global storage client."""
    global _storage_client
    if _storage_client is None:
        _storage_client = GCPStorageClient()
    return _storage_client
