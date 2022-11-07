"""
Script for downloading OpenStreetMap data to train deep learning models
the data can be downloaded using bbox coordinates and bbox coordinates for vector or raster files
Testing:
osmnx library has to install
Example command: python osm_data.py --bbox 30.33622 30.34384 78.03883 78.05268 --tags buildings
 --outfile ../data/dd_roads.geojson

The data has to clip with bounding box after downloading
"""

import osmnx as ox
import argparse


def download_osm(bbox, outfile, osm_tag):
    bbox_coord = tuple(bbox)
    # test  coordinates: north, south, east, west = 30.33622 ,30.34384,78.03883,78.05268
    north, south, east, west = bbox_coord[0], bbox_coord[1], bbox_coord[2], bbox_coord[3]
    print("Downloading: " + osm_tag)
    if osm_tag == "roads":
        tags = {'highway': True}
    elif osm_tag == "buildings":
        tags = {'building': True}
    gdf = ox.geometries_from_bbox(north, south, east, west, tags)
    gdf = gdf.apply(lambda c: c.astype(str) if c.name != 'geometry' else c, axis=0)
    gdf.to_file(outfile, driver='GeoJSON')
    print("Data downloaded and saved in " + outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs='+', type=float, help="Bounding box coordinates,"
                                                              " (north, south, east, west)")
    parser.add_argument("--tags", type=str, help="Feature types ex: buildings, roads")
    parser.add_argument("--outfile", type=str, help="Output filename with directory")
    args = parser.parse_args()

    download_osm(args.bbox, args.outfile, args.tags)
