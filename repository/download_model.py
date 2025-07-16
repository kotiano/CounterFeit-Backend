import os
import gdown
from logger_config import setup_logging

logger = setup_logging(__name__)

def model(gdrive_url: str, output_path: str) -> bool:
   
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Starting model download from Google Drive to {output_path}")
        gdown.download(gdrive_url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            logger.info(f"Model successfully downloaded to {output_path}")
            return True
        else:
            logger.error(f"Download failed: File not found at {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Model download error: {str(e)}", exc_info=True)
        return False