
import glob, os
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
from sklearn.metrics import r2_score


def list_files(path, ext, recursive=True):
    """
    Find file names recursively for a given string match

    :param path: the directory to search
    :param ext: the file extension to return
    :param recursive: search recursively or not, default to True
    :return:
    """
    if recursive is True:
        return glob.glob(os.path.join(path, '**', '*{}'.format(ext)), recursive=recursive)
    else:
        return glob.glob(os.path.join(path, '*{}'.format(ext)), recursive=recursive)


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


# ---------------------------------------------------------------------------
# Combined (mutually-exclusive) treatment variables per fire
# ---------------------------------------------------------------------------

# Default ACTIVITY groupings, using the harmonized (CFT) vocabulary produced by
# treatment_interactions.harmonize_twig_activity. Pile Burn already absorbs
# Machine/Hand Pile Burn and Jackpot Burn; Removal is standalone biomass removal
# (removal-that-is-a-cut is promoted to Mechanical upstream, so it counts in THIN).
THIN_ACTS    = {"Manual", "Mechanical"}
BURN_ACTS    = {"Broadcast Burn", "Pile Burn"}
REMOVAL_ACTS = {"Removal"}


def _make_valid(geom):
    """Return a valid version of ``geom`` (or None for null/empty)."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return None if (geom is None or geom.is_empty) else geom


def _union_geom(geoms):
    """Union a (possibly empty) iterable of geometries to a single *valid* geometry or None.

    Validity matters: area(A − B) == area(A) − area(A ∩ B) only holds for valid
    geometries, so the disjoint variables would not sum correctly otherwise.
    """
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None
    return _make_valid(shapely.unary_union(geoms))


def _safe_area_ac(geom):
    """Area in acres (assumes a metric CRS); 0.0 for None/empty."""
    if geom is None or geom.is_empty:
        return 0.0
    return geom.area / 4046.86


def _intersection_area_ac(a, b):
    """Acres of A ∩ B; 0.0 if either is None/empty."""
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b)
    return 0.0 if (inter is None or inter.is_empty) else inter.area / 4046.86


def combined_treatment_vars(trts_fire, buffer,
                            id_col="INCIDENT_ID", activity_col="ACTIVITY",
                            acres_col="fire_buffer_acres",
                            thin_acts=THIN_ACTS, burn_acts=BURN_ACTS,
                            removal_acts=REMOVAL_ACTS):
    """
    Build mutually-exclusive (disjoint) thin / burn / removal treatment variables
    per fire, so predictors do not reuse the same acres (avoids the collinearity
    that arises when a marginal total and its interaction both count the overlap).

    For each fire, union the harmonized treatment polygons into THIN, BURN, REMOVAL
    and FUELRED (=BURN or REMOVAL) footprints, then derive:

      Disjoint set A (thin x burn):    thin_only, burn_only, thin_x_burn
      Disjoint set B (thin x fuelred): thin_only_fr, fuelred_only, thin_x_fuelred
      Marginal totals (reference):     all_thin, all_burn, all_removal, all_fuelred
      Comparison only:                 thin_x_broadcast  (thin ∩ broadcast-burn-only)

    Within each disjoint set the pieces sum to the union footprint (no double count).
    Sets A and B are alternative model specifications, not for simultaneous use.

    The "only" pieces are derived **arithmetically** from the marginal and intersection
    areas (thin_only = all_thin − thin_x_burn, etc.) rather than via geometric
    ``difference``. That guarantees exact additivity and monotonicity
    (0 ≤ intersection ≤ each parent) instead of relying on floating-point
    ``difference`` results, which drift on large projected coordinates.

    Parameters
    ----------
    trts_fire : GeoDataFrame
        Treatments already clipped to each fire's buffer, carrying ``id_col`` and a
        harmonized ``activity_col``. Must be in a metric CRS (areas in m²).
    buffer : GeoDataFrame / DataFrame
        One row per fire with ``id_col`` and ``acres_col`` (buffer acres denominator).

    Returns
    -------
    DataFrame indexed by fire, with ``<var>_acres`` and ``<var>`` (percent of the
    fire buffer) columns for every variable above, plus ``fire_buffer_acres``.
    """
    buf_ac = buffer.set_index(id_col)[acres_col]
    rows = []
    for fid, sub in trts_fire.groupby(id_col):
        def U(acts):
            return _union_geom(sub.loc[sub[activity_col].isin(acts), "geometry"].values)
        thin_g  = U(thin_acts)
        burn_g  = U(burn_acts)
        rem_g   = U(removal_acts)
        bcast_g = U({"Broadcast Burn"})
        fuel_g  = _union_geom([burn_g, rem_g])

        all_thin    = _safe_area_ac(thin_g)
        all_burn    = _safe_area_ac(burn_g)
        all_removal = _safe_area_ac(rem_g)
        all_fuelred = _safe_area_ac(fuel_g)

        # intersection acres, clamped to their parents so the "only" pieces cannot
        # go negative and the disjoint sets sum exactly.
        tb  = min(_intersection_area_ac(thin_g, burn_g),  all_thin, all_burn)
        tf  = min(_intersection_area_ac(thin_g, fuel_g),  all_thin, all_fuelred)
        tf  = max(tf, tb)   # FUELRED ⊇ BURN, so thin∩fuelred ≥ thin∩burn
        tbc = min(_intersection_area_ac(thin_g, bcast_g), all_thin)
        tbc = min(tbc, tb)  # Broadcast ⊆ BURN, so thin∩broadcast ≤ thin∩burn

        vals = {
            "all_thin":         all_thin,
            "all_burn":         all_burn,
            "all_removal":      all_removal,
            "all_fuelred":      all_fuelred,
            # disjoint set A (thin x burn)
            "thin_only":        all_thin - tb,
            "burn_only":        all_burn - tb,
            "thin_x_burn":      tb,
            # disjoint set B (thin x fuels-reduction)
            "thin_only_fr":     all_thin - tf,
            "fuelred_only":     all_fuelred - tf,
            "thin_x_fuelred":   tf,
            # comparison only (vs old broadcast-only "Thin + Rx Fire")
            "thin_x_broadcast": tbc,
        }
        fa = float(buf_ac.get(fid, np.nan))
        row = {id_col: fid, acres_col: fa}
        for k, v in vals.items():
            row[f"{k}_acres"] = v
            row[k] = (v / fa * 100.0) if (fa and not np.isnan(fa)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)