import shutil
import requests
import os
import sys
import argparse

if __name__=='__main__':
    psr = argparse.ArgumentParser()
    psr.add_argument("--m3u8", type=str, required=True)
    args = psr.parse_args()

    if not os.path.exists(args.m3u8):
        sys.exit(f"File {args.m3u8} could not be found")

    # Parse playlist for filenames with ending .ts and put them into the list
    with open(args.m3u8, 'r') as playlist:
        ts_filenames = [line.rstrip() for line in playlist if line.rstrip().endswith('.ts')]

    print(ts_filenames)
    out_path = os.path.join(os.path.dirname(args.m3u8), os.path.basename(args.m3u8)+".ts")
    print(out_path)
    
    # open one ts_file from the list after another and append them to merged.ts
    with open(out_path, 'wb') as merged:
        for ts_file in ts_filenames:
            print(ts_file)
            try:
                rawImage = requests.get(ts_file, stream=True)
                print (rawImage.status_code)
                filename = ts_file.split("/")[-1]

                # save the image recieved into the file
                with open(filename, 'wb') as fd:
                    for chunk in rawImage.iter_content(chunk_size=1024):
                        fd.write(chunk)

                with open(filename, 'rb') as m1:
                    shutil.copyfileobj(m1, merged)

                # If file exists, delete it.
                if os.path.isfile(filename):
                    os.remove(filename)
                else:
                    print(f"Error: {filename} file not found")

            except requests.exceptions.RequestException:
                print(f'Invalid URL: "{ts_file}"')
                # at this point, you can continue the loop to the next URL if you want

    print(out_path)