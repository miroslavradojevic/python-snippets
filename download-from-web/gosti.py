#!/usr/bin/env python
# Download .mp3 episodes of Radio Belgrade show "Gosti iz proslosti"
# grab all mp3s and save them with parsed name and date to the output folder
from base64 import decode
import requests
import os
import sys
import time
import xml.dom.minidom
import eyed3
from urllib.parse import urlparse

url = "https://www.rts.rs/page/radio/sr/podcast/3590/gosti-iz-proslosti/audio.html"
# url results with xml that is further parsed

timestamp = time.strftime("%Y%m%d-%H%M%S")
out_dir = os.path.join("gosti_" + timestamp)
doc_path = "gosti_" + timestamp + ".xml"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

try:
    req = requests.get(url)
    req.raise_for_status()

    doc = xml.dom.minidom.parseString(req.text) # TODO check if it is valid XML

    items = doc.getElementsByTagName("item")
    print("found ", len(items), " items")

    for item in items:
        links = item.getElementsByTagName("link")
        if len(links) > 0:
            print(links[0].firstChild.data) # read element data value
            # get only filename of the .html https://bit.ly/2ZnqwK7
            a = urlparse(links[0].firstChild.data)
            out_fname_pname = os.path.basename(a.path).replace('.html', '')
        else:
            out_fname_pname = "NA"

        desriptions = item.getElementsByTagName("description")
        # https://docs.python.org/3/library/codecs.html#standard-encodings
        codec_name = "iso8859_16" # "cp1250" "iso8859_2" "cp852" "utf-8"
        
        dsc = desriptions[0].childNodes[0].data.encode(sys.stdout.encoding, errors='replace')
        dsc1 = dsc.decode(sys.stdout.encoding, errors='replace')

        enclosures = item.getElementsByTagName("enclosure")
        if len(enclosures) > 0:
            url_value = enclosures[0].attributes["url"].value # read attribute value
            print(url_value)
            if url_value.endswith('.mp3'):
                url_elements = urlparse(url_value).path.split('/')

                if len(url_elements) >= 5:
                    out_fname_date = ''.join(url_elements[-5:-2]) # https://bit.ly/3e6mXMk
                else:
                    out_fname_date = "NA"
                
                out_file = os.path.join(out_dir, out_fname_date + "_" + out_fname_pname + ".mp3")

                # save mp3 file from url_value to out_file
                # https://dzone.com/articles/simple-examples-of-downloading-files-using-python
                print("saving... ", end='')
                try:
                    req = requests.get(url_value)
                    req.raise_for_status()

                    open(out_file, 'wb').write(req.content)
                    print("saved to " + out_file)
                    # add description
                    mp3 = eyed3.load(out_file)
                    if mp3 is not None:
                        mp3.initTag() # mp3.tag.artist, mp3.tag.comments
                        mp3.tag.comments.set(dsc1)
                        mp3.tag.save()
                        print(dsc1)

                except requests.exceptions.HTTPError as err:
                    print(err)
                    # raise SystemExit(err)

        print("\n")

    # save rss xml
    with open(os.path.join(out_dir, doc_path), "w", encoding="utf-8") as f:
        f.write(doc.toprettyxml())
    print(os.path.join(out_dir, doc_path))

except requests.exceptions.HTTPError as err:
    print(err)
    # raise SystemExit(err)
