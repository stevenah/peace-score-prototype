import {
  S3Client,
  GetObjectCommand,
  HeadObjectCommand,
  PutObjectCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

let _client: S3Client | null = null;

function getClient(): S3Client {
  if (!_client) {
    // S3_ENDPOINT points at an S3-compatible server (e.g. MinIO on
    // http://localhost:9000) for fully local development. Unset => real AWS.
    const endpoint = process.env.S3_ENDPOINT;
    _client = new S3Client({
      region: process.env.AWS_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
      },
      ...(endpoint ? { endpoint, forcePathStyle: true } : {}),
    });
  }
  return _client;
}

export async function getPresignedUrl(key: string, expiresIn = 3600) {
  const bucket = process.env.S3_BUCKET_NAME;
  if (!bucket) {
    throw new Error("S3_BUCKET_NAME not configured");
  }
  return getSignedUrl(
    getClient(),
    new GetObjectCommand({ Bucket: bucket, Key: key }),
    { expiresIn },
  );
}

export async function getObjectStream(key: string, range?: string) {
  const bucket = process.env.S3_BUCKET_NAME;
  if (!bucket) {
    throw new Error("S3_BUCKET_NAME not configured");
  }
  const command = new GetObjectCommand({
    Bucket: bucket,
    Key: key,
    ...(range ? { Range: range } : {}),
  });
  return getClient().send(command);
}

export async function getObjectHead(key: string) {
  const bucket = process.env.S3_BUCKET_NAME;
  if (!bucket) {
    throw new Error("S3_BUCKET_NAME not configured");
  }
  const command = new HeadObjectCommand({ Bucket: bucket, Key: key });
  return getClient().send(command);
}

export async function getPresignedPutUrl(
  key: string,
  contentType: string,
  expiresIn = 3600,
) {
  const bucket = process.env.S3_BUCKET_NAME;
  if (!bucket) {
    throw new Error("S3_BUCKET_NAME not configured");
  }
  return getSignedUrl(
    getClient(),
    new PutObjectCommand({ Bucket: bucket, Key: key, ContentType: contentType }),
    { expiresIn },
  );
}
