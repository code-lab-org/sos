import boto3

# def upload_file_to_s3(local_path, bucket_name, s3_key):
#     s3 = boto3.client("s3")
#     s3.upload_file(local_path, bucket_name, s3_key)
#     print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")

# if __name__ == "__main__":
#     local_path = "sat000038337.txt"
#     bucket_name = "snow-observing-systems"
#     s3_key = "inputs/satellite/sat000038337.txt" 

#     upload_file_to_s3(local_path, bucket_name, s3_key)

def fetch_tle_lines_from_s3(bucket="snow-observing-systems", key="inputs/satellite/sat000038337.txt"):
    import boto3

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')

    # Split lines and strip whitespace
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    tle_lines = []
    i = 0
    while i < len(lines) - 2:
        if lines[i].startswith("1 ") and lines[i+1].startswith("2 "):
            # Already TLE lines without name, skip i+=2
            tle_lines.append(lines[i])
            tle_lines.append(lines[i+1])
            i += 2
        elif lines[i].startswith("GCOM") and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            # Found a name + valid TLE pair
            tle_lines.append(lines[i+1])  # Line 1
            tle_lines.append(lines[i+2])  # Line 2
            i += 3
        else:
            i += 1  # skip garbage or malformed line

    return tle_lines


tle=fetch_tle_lines_from_s3()
print(tle)