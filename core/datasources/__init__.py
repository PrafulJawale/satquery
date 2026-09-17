"""Phase 8 -- external data sources.

    core/datasources/base.py           windowed fetch + cache + provenance
    core/datasources/worldcover.py     ESA WorldCover 2021 v200 (10 m, categorical)
    core/datasources/soilgrids.py      ISRIC SoilGrids 2.0 (250 m, model output)
    core/datasources/worldclim.py      WorldClim 2.1 monthly climatologies (~1 km)
    core/datasources/copernicus_dem.py Copernicus DEM GLO-30 (30 m)

Rules every source follows:
    * read only the window that covers the analysis grid -- never a global raster
    * continuous layers are resampled bilinearly, categorical ones nearest
    * every layer is returned with a machine-readable provenance record
    * a failure is reported as missing data, never as a value
"""

from . import base, copernicus_dem, soilgrids, worldclim, worldcover

__all__ = ["base", "copernicus_dem", "soilgrids", "worldclim", "worldcover"]
