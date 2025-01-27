import json
import boto3
import logging
from datetime import datetime
import urllib.parse

# Initialize clients
s3 = boto3.client('s3')

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("Received event: " + json.dumps(event, indent=2))
    try:
        for record in event['Records']:
            source_bucket = record['s3']['bucket']['name']
            source_key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
            
            # Prevent processing already renamed files
            if source_key.startswith("renamed/"):
                logger.info(f"File {source_key} is already renamed. Skipping.")
                continue
            
            # Extract filename and extension
            filename = source_key.split('/')[-1]
            name, extension = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            
            # Generate timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            
            # Create new filename
            new_filename = f"{name}_{timestamp}.{extension}" if extension else f"{name}_{timestamp}"
            
            # Define destination key
            destination_key = f"renamed/{new_filename}"
            
            logger.info(f"Renaming {source_key} to {destination_key}")
            
            # Copy the object to the new key
            copy_source = {
                'Bucket': source_bucket,
                'Key': source_key
            }
            
            s3.copy_object(
                Bucket=source_bucket,
                CopySource=copy_source,
                Key=destination_key
            )
            
            # Delete the original object
            s3.delete_object(
                Bucket=source_bucket,
                Key=source_key
            )
            
            logger.info(f"Successfully renamed {source_key} to {destination_key}")
            
    except Exception as e:
        logger.error(f"Error processing object: {e}")
        raise e

    return {
        'statusCode': 200,
        'body': json.dumps('File renamed successfully!')
    }
