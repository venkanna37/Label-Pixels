"""
Rasterising vector dataset  or Creating label data
Example added in the README file

keywords:
Raster: Image/Grid of Pixels (format: tif)
Vector: Shapes/List of coordinates (formats: .shp, .geojson)

"""

import argparse
import gdal
import glob
import numpy as np
import ogr
import os
import osr
import sys


def rasterize(raster_dir, raster_format, vector_dir, vector_format,
              output_dir, buffer, buffer_atr, label_atr):
    # Load all Raster and Vector files
    raster_files = sorted(glob.glob(os.path.join(raster_dir, "*." + raster_format)))
    vector_files = sorted(glob.glob(os.path.join(vector_dir, "*." + vector_format)))

    assert len(raster_files) != 0, "There are no Raster files in the directory"
    assert len(vector_files) != 0, "There are no Vector files in the directory"
    assert len(vector_files) == len(raster_files), "Raster and Vector files are not equal"

    print("Loaded " + str(len(vector_files)) + " Vector and Raster files " + "\n")

    # Check raster and vector file names ( is same or not)
    r_files, v_files, r_not_matched_files, v_not_matched_files = check_filenames(raster_files, vector_files,
                                                                                 raster_format, vector_format)

    assert len(r_files) != 0 and len(v_files) != 0 and len(v_files) == len(r_files)
    print("Rasterizing " + str(len(r_files)) + " file(s)" + "\n")

    for r, v in zip(r_files, v_files):
        # Loop through raster and vector files and rasterize
        # print(r, v)
        raster_layer = gdal.Open(r)

        if vector_format == "geojson":
            driver = ogr.GetDriverByName('GeoJSON')
        elif vector_format == "shp":
            driver = ogr.GetDriverByName('ESRI Shapefile')
        else:
            print("Check file format, This tool Rasterize only geojson and shapefiles")

        vector_layer = driver.Open(v, 0)
        # print(vector_layer, raster_layer)

        # (1)Check Projection of vector and raster datasets
        ds = vector_layer.GetLayer().GetSpatialRef()
        # print(type(ds))
        # print(raster_layer.GetProjection())
        vector_epsg = ds.GetAttrValue('AUTHORITY', 1)
        proj = osr.SpatialReference(wkt=raster_layer.GetProjection())
        # print(proj)
        raster_epsg = proj.GetAttrValue('AUTHORITY', 1)

        # print(vector_epsg, raster_epsg)
        assert raster_epsg is not None, "Projection is not defined for Raster file"
        assert vector_epsg is not None, "Projection is not defined for Raster file"
        assert vector_epsg == raster_epsg, "The projections of Vector and Raster files are not same"

        # (2) Creating the destination raster data source
        fn = os.path.basename(r)
        output_file =os.path.join(output_dir, fn)
        # print(output_file)
        ref_image = raster_layer
        gt = ref_image.GetGeoTransform()
        proj = ref_image.GetProjection()
        [cols, rows] = np.array(ref_image.GetRasterBand(1).ReadAsArray()).shape
        target_ds = gdal.GetDriverByName('GTiff').Create(output_file, rows, cols, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(gt)
        target_ds.SetProjection(proj)
        target_ds.GetRasterBand(1)

        v_layer = vector_layer.GetLayer()
        feature = v_layer.GetNextFeature()
        geom = feature.GetGeometryRef()

        if geom.GetGeometryType() == ogr.wkbLineString:
            rasterize_line(v_layer, buffer, driver, vector_epsg, target_ds, label_atr, buffer_atr)
        elif geom.GetGeometryType() == ogr.wkbPolygon:
            rasterize_polygon(v_layer, target_ds, label_atr)
        else:
            print("The Geometry type is neither Polygon nor Line")


# print("Rasterize N number of file" + "\n" + "Failed to rasterize N number of files")


def rasterize_line(v_layer, buffer, driver, vector_epsg, target_ds, label_atr, buffer_atr):

    """
    Rasterizing line feature with taking buffer from command line or attributes
    :param r_layer: Raster layer
    :param v_layer: Vector layer
    :return: Rasterized layer
    """

    # print(geom.GetGeometryType(), ogr.wkbLineString)
    #  Creating polygon feature from line feature with buffer
    file = '../log_dir/buffer.shp'
    if os.path.exists(file):
        driver.DeleteDataSource(file)
    ds = driver.CreateDataSource(file)
    if ds is None:
        print('Could not create file')
        sys.exit(1)

    geosr = osr.SpatialReference()
    geosr.ImportFromEPSG(int(vector_epsg))
    lyrBuffer = ds.CreateLayer('buffer', geom_type=ogr.wkbPolygon, srs=geosr)
    featureDefn = lyrBuffer.GetLayerDefn()
    feature = v_layer.GetNextFeature()
    fieldDefn = feature.GetFieldDefnRef(label_atr)
    lyrBuffer.CreateField(fieldDefn)

    while feature:
        if buffer_atr:
            try:
                buffer_width = (float(feature.GetField(buffer_atr)) * buffer)
            except ValueError:
                print("The type of attribute value should be integer or float")
                sys.exit()
            except KeyError:
                print("Attribute does not exist in shapefile attributes")
                sys.exit()
        else:
            buffer_width = buffer
        label_atr_value = feature.GetField(label_atr)
        geomTest = feature.GetGeometryRef()
        if vector_epsg == "4326":
            # Getting spatial reference of input raster
            utmsr = osr.SpatialReference()
            utmsr.ImportFromEPSG(32643)
            # OSR transformation
            geosr_utmsr = osr.CoordinateTransformation(geosr, utmsr)
            utmsr_geosr = osr.CoordinateTransformation(utmsr, geosr)
            geomTest.Transform(geosr_utmsr)
            geomBuffer = geomTest.Buffer(buffer_width)
            geomBuffer.Transform(utmsr_geosr)
        else:
            geomBuffer = geomTest.Buffer(buffer_width)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        outFeature.SetField(label_atr, label_atr_value)
        lyrBuffer.CreateFeature(outFeature)
        outFeature.Destroy()
        feature = v_layer.GetNextFeature()
    if label_atr:
        label_atri = ["ATTRIBUTE=" + label_atr]
        # gdal.RasterizeLayer(ds,bands,layer,burn_values, options = ["BURN_VALUE_FROM=Z"])
        gdal.RasterizeLayer(target_ds, [1], lyrBuffer, options=label_atri)
        target_ds = None
    else:
        gdal.RasterizeLayer(target_ds, [1], v_layer)
        target_ds = None
    ds = None


def rasterize_polygon(v_layer, target_ds, label_atr):
    """
    Rasterizing polygon data
    :param v_file:
    :param r_file:
    :return:
    """

    if label_atr:
        label_atri = ["ATTRIBUTE=" + label_atr]
        # gdal.RasterizeLayer(ds,bands,layer,burn_values, options = ["BURN_VALUE_FROM=Z"])
        gdal.RasterizeLayer(target_ds, [1], v_layer, options=label_atri)
        target_ds = None
    else:
        gdal.RasterizeLayer(target_ds, [1], v_layer)
        target_ds = None


def check_filenames(raster_files, vector_files, raster_format, vector_format):
    """
    Function for checking raster and vector filenames
    Assuming there are no repeated files in directories
    :param raster_files: List of raster file paths
    :param vector_files: List of vector file paths
    :return: Returns files those are matching and not matching
    """
    r_files = []
    v_files = []
    r_not_matched_files = []
    v_not_matched_files = []
    for i, j in zip(raster_files, vector_files):
        _, raster_filename = os.path.split(i)
        _, vector_filename = os.path.split(j)
        raster_filename = raster_filename[: -len(raster_format)-1]
        vector_filename = vector_filename[: -len(vector_format)-1]
        # print(raster_filename, vector_filename)
        if raster_filename != vector_filename:
            r_not_matched_files.append(i)
            v_not_matched_files.append(j)
        else:
            r_files.append(i)
            v_files.append(j)
    print("Number of matched file(s): " + str(len(r_files)))
    print("Number of unmatched file(s): " + str(len(r_not_matched_files)) + "\n")
    return r_files, v_files, r_not_matched_files, v_not_matched_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster_dir", type=str, help="Directory containing Raster image/images with projection")
    parser.add_argument("--vector_dir", type=str, help="Directory containing Vector files with the same projection"
                                                       " as Raster data and name of the vector file name should"
                                                       " be same the respective raster file name")
    parser.add_argument("--raster_format", type=str, help=" Raster format of the Image/Images", default="tif")
    parser.add_argument("--vector_format", type=str, help=" Raster format of the Image/Images", default="geojson")
    parser.add_argument("--buffer", type=int, help="Buffer width of line feature (Only for Lines)", default=1)
    parser.add_argument("--output_dir", type=str, help="Output image name with directory")
    parser.add_argument("--buffer_atr", type=str, help="Attribute from the vector file, This attribute can be"
                                                       " buffer Width or It multiplies with buffer", default=None)
    parser.add_argument("--label_atr", type=str, help="Attribute from the vector file to assign label to pixel."
                                                      "Not required for single class")
    args = parser.parse_args()

    rasterize(args.raster_dir, args.raster_format, args.vector_dir, args.vector_format,
    args.output_dir, args.buffer, args.buffer_atr, args.label_atr)
