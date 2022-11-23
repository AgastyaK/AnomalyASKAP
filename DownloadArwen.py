# Author: Matt Austin

import logging
import pathlib

import requests


logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)


DOWNLOAD_DIRECTORY = pathlib.Path("PLOTS")


with open(
    "ampantbeam_waterfall_autoscale.txt", encoding="utf-8", mode="r"
) as url_list:
    for url in url_list.readlines():
        url = url.strip()
        sbid, filename = url.split("/")[-2:]

        # Construct local path
        local_path = pathlib.Path(
            DOWNLOAD_DIRECTORY / pathlib.Path(sbid) / pathlib.Path(filename)
        )

        # Only proceed if the local_path doesn't already exist
        if not local_path.exists():

            # Create sbid directory if it doesn't exist
            if not local_path.parent.exists():
                local_path.parent.mkdir(parents=True)

            # Fetch file
            print(url)
            logger.info(f"Downloading {url} to {local_path}...")
            response = requests.get(url)
            response.raise_for_status()

            # Write file
            with local_path.open(mode="wb") as local_file:
                local_file.write(response.content)
