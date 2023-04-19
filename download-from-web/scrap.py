import shutil
import requests
import os
# from urllib3.exceptions import NewConnectionError

# Parse playlist for filenames with ending .ts and put them into the list ts_filenames
with open('/home/miro/Downloads/201237597_.m3u8', 'r') as playlist:
    ts_filenames = [line.rstrip() for line in playlist
                    if line.rstrip().endswith('.ts')]

    print(ts_filenames)

    # open one ts_file from the list after another and append them to merged.ts
    with open('201237597_.ts', 'wb') as merged:
        for ts_file in ts_filenames:
            print(ts_file)
            try:
                # resp = requests.get(url)
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
                    # If it fails, inform the user.
                    print("Error: %s file not found" % filename)

            except requests.exceptions.RequestException:
                print(f'Invalid URL: "{ts_file}"')
                # at this point, you can continue the loop to the next URL if you want

            
