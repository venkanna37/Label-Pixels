import ogr
import osr
import gdal
import numpy as np
import os
import glob
import argparse


def rasterize(args):
    raster = args.raster
    vector = args.vector
    shpdriver = ogr.GetDriverByName('ESRI Shapefile')

    # Projections to add buffer in meters
    geosr = osr.SpatialReference()
    geosr.ImportFromEPSG(4326)
    utmsr = osr.SpatialReference()
    utmsr.ImportFromEPSG(32643)
    geosr2utmsr = osr.CoordinateTransformation(geosr, utmsr)
    utmsr2geosr = osr.CoordinateTransformation(utmsr, geosr)

    base = os.path.basename(vector)
    file_name, ext = os.path.splitext(base)
    print('Rasterizing:  ' + file_name)
    test = ogr.Open(vector, 0)
    lyrTest = test.GetLayer()
    v_proj = lyrTest.GetSpatialRef()

    # Buffer
    ds = shpdriver.CreateDataSource('buffer.shp')  # how to create buffer without creating shp
    shp_filename = shp_directory + file_name + '.shp'
    ds_1 = shpdriver.CreateDataSource(shp_filename)
    lyrBuffer = ds.CreateLayer('buffer', geom_type=ogr.wkbPolygon, srs=utmsr)
    lyrBuffer1 = ds_1.CreateLayer(file_name, geom_type=ogr.wkbPolygon, srs=v_proj)
    featureDefn = lyrBuffer.GetLayerDefn()

    featureTest = lyrTest.GetNextFeature()
    while featureTest:
        geomTest = featureTest.GetGeometryRef()
        geomTest.Transform(geosr2utmsr)
        if args.attribute:
            buffer_width = (float(featureTest.GetField(args.attribute)) * args.buffer)
        else:
            buffer_width = args.buffer
        geomBuffer = geomTest.Buffer(buffer_width)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        lyrBuffer.CreateFeature(outFeature)
        outFeature.Destroy()
        featureTest = lyrTest.GetNextFeature()

    buffer_featureTest = lyrBuffer.GetNextFeature()
    while buffer_featureTest:
        geomTest = buffer_featureTest.GetGeometryRef()
        geomTest.Transform(utmsr2geosr)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomTest)
        lyrBuffer1.CreateFeature(outFeature)
        outFeature.Destroy()
        buffer_featureTest = lyrBuffer.GetNextFeature()

    # Rasterize
    # reference_image = 'SN5_roads_train_AOI_8_Mumbai_PS-RGB_chip0.tif'
    raster_path = args.file_name + '.tif'
    ref_image = gdal.Open(reference_image)
    gt = ref_image.GetGeoTransform()
    proj = ref_image.GetProjection()

    # 2) Creating the destination raster data source
    pixelWidth = pixelHeight = gt[1]  # depending how fine you want your raster ##COMMENT 1
    [cols, rows] = np.array(ref_image.GetRasterBand(1).ReadAsArray()).shape
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_path, cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(gt)

    # 5) Adding a spatial reference
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    # gdal.RasterizeLayer(ds,bands,layer,burn_values, options = ["BURN_VALUE_FROM=Z"])
    gdal.RasterizeLayer(target_ds, [1], lyrBuffer1)
    # print(target_ds)
    target_ds = None

    # Deleting buffer shapefie with UTM projection
    os.remove("buffer.shp")
    os.remove("buffer.dbf")
    os.remove("buffer.shx")
    os.remove("buffer.prj")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", type=str, help="Raster file path")
    parser.add_argument("--vector", type=str, help="Vector file path")
    parser.add_argument("--buffer", type=int, help="Buffer width of line feature")
    parser.add_argument("--output_file", type=str, help="Output image name")
    parser.add_argument("--attribute", type=str, help="Attribute from the vector file")
    args = parser.parse_args()
    rasterize(args)
