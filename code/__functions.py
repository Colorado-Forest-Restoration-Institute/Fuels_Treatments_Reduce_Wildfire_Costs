
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from shapely.geometry import shape

def fetch_nsi_fips(fips_list, timeout=120):
    nsi_results = []
    failed_fips = []

    for fips in tqdm(fips_list, desc="Downloading NSI by county FIPS"):
        try:
            url = f"https://nsi.sec.usace.army.mil/nsiapi/structures?fips={fips}&fmt=fc"
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            features = response.json().get("features", [])
            if features:
                gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                gdf["FIPS"] = fips
                nsi_results.append(gdf)
            else:
                print(f"No structures returned for FIPS {fips}")
                failed_fips.append(fips)

        except Exception as e:
            print(f"Failed to retrieve FIPS {fips}: {e}")
            failed_fips.append(fips)

    return nsi_results, failed_fips


def get_feature_service_gdf(url, geo=None, qry='1=1', layer=0):
    """
    Description
    GeoDataFrame from a Feature Service from url and optional bounding geometry and where clause

    Parameters
    ----------
    url : STRING
        Base url for the feature service.
    geo : OBJECT, optional
        Bounding box string, shapely polygon, geodataframe, or geoseries. The default is ''.
    qry : TYPE, optional
        Where clause used to subset the data. The default is '1=1'.
    layer : TYPE, optional
        Extent of the Feature Service. The default is 0.

    Returns
    -------
    LIST
        GeoDataFrame of features.

    """

    # Gather info from the Feature Service
    s_info = requests.get(url + '?f=pjson').json()  # json metadata
    srn = s_info['spatialReference']['wkid']  # spatial reference
    sr = 'EPSG:' + str(srn)
    # print(f"Feature service CRS: {sr}")

    # Handle the bounding geometry if needed
    # If no bounding geometry is provided, returns all
    if geo is not None:
        if isinstance(geo, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geo = geo.to_crs(sr).total_bounds
        elif isinstance(geo, shapely.geometry.base.BaseGeometry):
            geo = gpd.GeoSeries([geo], crs=sr).to_crs(sr).total_bounds
        elif isinstance(geo, (list, tuple, np.ndarray)) and len(geo) == 4:
            geo = np.array(geo)
        else:
            raise ValueError("Invalid geometry input.")

        # Sanity check bounds
        if not np.all(np.isfinite(geo)):
            raise ValueError(f"Non-finite geometry bounds encountered: {geo}")

        geo = ','.join(geo.astype(str))
    else:
        geo = None

    # Extract the correct URL for the Feature Service layer
    url1 = url + '/' + str(layer)  # adds the layer identifier (eg, 0)
    # Get the Feature Service metadata information
    l_info = requests.get(url1 + '?f=pjson').json()
    maxrcn = l_info['maxRecordCount']  # number of records the service allows per query
    if maxrcn > 100: maxrcn = 100  # used to subset ids so query is not so long
    url2 = url1 + '/query?'  # base URL for service requests

    # Get a list of Object IDs (OIDs) for features matching the filter
    o_info = requests.get(
        url2, {
            'where': qry,
            'geometry': geo,
            'geometryType': 'esriGeometryEnvelope',
            'returnIdsOnly': 'True',
            'f': 'pjson'
        }).json()

    # Gather the OIDs
    oid_name = o_info['objectIdFieldName']
    oids = o_info['objectIds']
    numrec = len(oids)  # number of records returned

    # Gather the list of features
    fslist = []
    for i in range(0, numrec, maxrcn):
        objectIds = oids[i:i + maxrcn]
        idstr = oid_name + ' in (' + str(objectIds)[1:-1] + ')'
        prm = {
            'where': idstr,
            'outFields': '*',
            'returnGeometry': 'true',
            'outSR': srn,
            'f': 'pgeojson',
        }
        response = requests.get(url2, prm)

        # Fallback to standard geojson if pgeojson fails
        try:
            ftrs = response.json()['features']
        except (requests.exceptions.JSONDecodeError, KeyError):
            prm['f'] = 'geojson'
            response = requests.get(url2, prm)
            try:
                ftrs = response.json()['features']
            except Exception as e:
                raise RuntimeError(
                    f"Failed to retrieve features from {url2}\nResponse text: {response.text[:300]}...") from e

        # convert features to a geodataframe
        ftrs_gdf = gpd.GeoDataFrame.from_features(ftrs, crs=sr)
        ftrs_gdf = ftrs_gdf.dropna(axis=1, how='all')  # remove all-NA columns
        fslist.append(ftrs_gdf)

    fslist = [df for df in fslist if not df.empty]  # remove empty frames

    if fslist:
        return gpd.pd.concat(fslist, ignore_index=True)
    else:
        return gpd.GeoDataFrame()  # return empty GeoDataFrame if no features